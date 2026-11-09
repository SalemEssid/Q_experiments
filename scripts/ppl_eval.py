"""Step 2: WikiText-2 perplexity, the standard accuracy metric in the GPTQ, AWQ, QuaRot and SpinQuant papers.

Protocol (the same one those papers use, so our numbers are comparable to theirs):
  1. Join the WikiText-2 test split into one long text and tokenize it once.
  2. Cut the tokens into non-overlapping windows of 2048 tokens.
  3. In each window the model predicts every token from the tokens before it. The loss is the
     average negative log-likelihood (NLL) of the correct next tokens.
  4. Perplexity = exp(mean NLL over all windows). Lower is better.

Usage (from ~/Q_experiments):
  python scripts/ppl_eval.py --model meta-llama/Llama-3.2-1B
"""
import argparse
import json
import time
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Hugging Face model ID or local folder")
parser.add_argument("--seqlen", type=int, default=2048, help="window length in tokens")
parser.add_argument("--out-dir", default="results/ppl")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
model.eval()

test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
n_windows = ids.shape[1] // args.seqlen  # the leftover tokens at the end are dropped
print(f"{ids.shape[1]:,} tokens -> {n_windows} windows of {args.seqlen} tokens")

nlls = []
start = time.perf_counter()
with torch.inference_mode():  # no gradients needed: we only evaluate
    for i in range(n_windows):
        window = ids[:, i * args.seqlen : (i + 1) * args.seqlen].to(model.device)
        # labels=window makes the model shift the tokens by one and compute the next-token loss.
        loss = model(window, labels=window, use_cache=False).loss
        nlls.append(loss.float())
        if (i + 1) % 20 == 0:
            running = torch.exp(torch.stack(nlls).mean()).item()
            print(f"  {i + 1}/{n_windows} windows, running PPL {running:.3f}")
elapsed = time.perf_counter() - start
ppl = torch.exp(torch.stack(nlls).mean()).item()

result = {
    "model": args.model,
    "dataset": "wikitext-2-raw-v1 (test)",
    "seqlen": args.seqlen,
    "n_windows": n_windows,
    "ppl": round(ppl, 4),
    "eval_seconds": round(elapsed, 1),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
}
out = Path(args.out_dir) / f"{Path(args.model).name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("Saved to", out)
