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

# W8A8 check: compressed-tensors gives every quantized Linear layer a forward function that rounds its
# input activations to INT8 before the matrix multiply (fake quantization). Prove it is active by scoring
# the first window with activation quantization on and off: the losses must differ. "Off" keeps the
# 8-bit weights, so it also shows how much of the error comes from the weights alone.
act_layers = [
    m for m in model.modules() if getattr(getattr(m, "quantization_scheme", None), "input_activations", None)
]
act_check = None
if act_layers:
    first = ids[:, : args.seqlen].to(model.device)
    with torch.inference_mode():
        loss_on = model(first, labels=first, use_cache=False).loss.item()
        for m in act_layers:
            m.quantization_enabled = False
        loss_off = model(first, labels=first, use_cache=False).loss.item()
        for m in act_layers:
            m.quantization_enabled = True
    act_check = {"layers": len(act_layers), "first_window_loss_act_on": round(loss_on, 5),
                 "first_window_loss_act_off": round(loss_off, 5)}
    print(f"Activation quantization in {len(act_layers)} layers. First-window loss: "
          f"{loss_on:.5f} with it, {loss_off:.5f} without it (must differ)")

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
    "activation_quant_check": act_check,  # None for BF16 and weight-only (W4A16) models
    "eval_seconds": round(elapsed, 1),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
}
out = Path(args.out_dir) / f"{Path(args.model).name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("Saved to", out)
