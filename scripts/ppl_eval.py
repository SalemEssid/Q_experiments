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
parser.add_argument(
    "--weights-only",
    action="store_true",
    help="W8A8 models only: turn activation quantization off for the whole evaluation, so only the error "
    "of the 8-bit weights is measured. Saved to results/ppl_weights_only/",
)
parser.add_argument(
    "--act-off-in",
    default=None,
    help="W8A8 models only: comma-separated parts of layer names, e.g. down_proj or q_proj,k_proj,v_proj. "
    "Activation quantization is turned off only in those layers, to find where the activation error comes "
    "from. Saved to results/ppl_act_diag/",
)
parser.add_argument("--out-dir", default=None, help="default: results/ppl, or the folders named above")
args = parser.parse_args()
if args.weights_only and args.act_off_in:
    parser.error("use either --weights-only or --act-off-in")
out_dir = args.out_dir or (
    "results/ppl_weights_only" if args.weights_only else "results/ppl_act_diag" if args.act_off_in else "results/ppl"
)

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
act_layers = {
    name: m
    for name, m in model.named_modules()
    if getattr(getattr(m, "quantization_scheme", None), "input_activations", None)
}
act_check = None
if act_layers:
    first = ids[:, : args.seqlen].to(model.device)
    with torch.inference_mode():
        loss_on = model(first, labels=first, use_cache=False).loss.item()
        for m in act_layers.values():
            m.quantization_enabled = False
        loss_off = model(first, labels=first, use_cache=False).loss.item()
        for m in act_layers.values():
            m.quantization_enabled = True
    act_check = {"layers": len(act_layers), "first_window_loss_act_on": round(loss_on, 5),
                 "first_window_loss_act_off": round(loss_off, 5)}
    print(f"Activation quantization in {len(act_layers)} layers. First-window loss: "
          f"{loss_on:.5f} with it, {loss_off:.5f} without it (must differ)")

if (args.weights_only or args.act_off_in) and not act_layers:
    raise SystemExit("--weights-only and --act-off-in need a model with activation quantization (W8A8)")
if args.weights_only:
    for m in act_layers.values():
        m.quantization_enabled = False
    print("Activation quantization OFF for this evaluation: only the weight error is measured")
act_off_layers = None
if args.act_off_in:
    parts = args.act_off_in.split(",")
    act_off_layers = [name for name in act_layers if any(p in name for p in parts)]
    if not act_off_layers:
        raise SystemExit(f"no quantized layer name contains any of {parts}")
    for name in act_off_layers:
        act_layers[name].quantization_enabled = False
    print(f"Activation quantization OFF in {len(act_off_layers)} of {len(act_layers)} layers ({args.act_off_in})")

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
    "weights_only": args.weights_only,
    "act_off_in": args.act_off_in,
    "act_off_layers": len(act_off_layers) if act_off_layers else None,
    "eval_seconds": round(elapsed, 1),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
}
stem = Path(args.model).name
if args.act_off_in:
    stem += "-actoff-" + args.act_off_in.replace(",", "+")
out = Path(out_dir) / f"{stem}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("Saved to", out)
