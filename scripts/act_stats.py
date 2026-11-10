"""Why some layer inputs are hard to quantize: activation statistics on one WikiText-2 window.

For every input of the Linear layers in each block (q/k/v share one input, gate/up share one), it measures:
  max          the largest |value| in the window, and the token where it occurs
  typical      the median over tokens of each token's largest |value| (the range of an ordinary token)
  max/typical  what one static INT8 scale per layer does to ordinary tokens. The scale is max/127, so an
               ordinary token only uses about 127/(max/typical) levels, and any value below max/254 rounds
               to zero. Above about 254, ordinary tokens become all zeros.
  without top  the same ratio when the single most extreme token is left out. If it drops a lot, the problem
               is ONE token (a "massive activation"), not a whole channel. A per-token scale handles that
               (that token gets its own scale); a per-channel smoothing like SmoothQuant cannot.
Quantized checkpoints are measured with quantization switched off, so a SmoothQuant model shows its smoothed
activations in BF16.

Usage (from ~/Q_experiments, quant venv):
  python scripts/act_stats.py --model meta-llama/Llama-3.2-1B
  python scripts/act_stats.py --model models/Llama-3.2-1B-SQ-W8A8
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUTS = {"q_proj": "q/k/v", "o_proj": "o_proj", "gate_proj": "gate/up", "down_proj": "down_proj"}

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--seqlen", type=int, default=2048)
parser.add_argument("--window", type=int, default=1, help="which 2048-token window (1 = the second: no BOS token)")
parser.add_argument("--detail", default="down_proj", choices=list(INPUTS), help="input type shown in detail")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
model.eval()
quantized = [m for m in model.modules() if getattr(m, "quantization_scheme", None) is not None]
for m in quantized:
    m.quantization_enabled = False
if quantized:
    print(f"Quantization switched off in {len(quantized)} layers: measuring the BF16 activations of this checkpoint")

test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
window = ids[:, args.window * args.seqlen : (args.window + 1) * args.seqlen].to(model.device)

stats = {}  # (block index, input type) -> numbers


def make_hook(block, kind):
    def hook(module, inputs):
        x = inputs[0][0].float().abs()  # tokens x channels (batch of 1)
        token_max = x.amax(dim=1)  # largest |value| of each token
        top = int(token_max.argmax())
        typical = token_max.median().item()
        without_top = torch.cat([token_max[:top], token_max[top + 1 :]]).max().item()
        stats[(block, kind)] = {
            "max": token_max[top].item(),
            "position": top,
            "token": tokenizer.decode([window[0, top].item()]),
            "typical": typical,
            "ratio": token_max[top].item() / typical,
            "ratio_without_top": without_top / typical,
        }

    return hook


for name, module in model.named_modules():
    parts = name.split(".")
    if parts[-1] in INPUTS and "layers" in parts:
        block = int(parts[parts.index("layers") + 1])
        module.register_forward_pre_hook(make_hook(block, parts[-1]))

with torch.inference_mode():
    model(window, use_cache=False)

n_blocks = max(b for b, _ in stats) + 1
print(f"\n{args.model}, WikiText-2 test window {args.window} ({args.seqlen} tokens)")
print("\nmax / typical token range of each layer input (above ~254, one static INT8 scale rounds ordinary tokens to 0)")
print("| Block | " + " | ".join(INPUTS.values()) + " |")
print("|---|" + "---|" * len(INPUTS))
for b in range(n_blocks):
    print(f"| {b} | " + " | ".join(f"{stats[(b, k)]['ratio']:.0f}" for k in INPUTS) + " |")

print(f"\n{INPUTS[args.detail]} input in detail")
print("| Block | Max abs value | At token (position) | Typical token max | Max / typical | Without that token |")
print("|---|---|---|---|---|---|")
for b in range(n_blocks):
    s = stats[(b, args.detail)]
    print(
        f"| {b} | {s['max']:.1f} | {s['token']!r} ({s['position']}) | {s['typical']:.2f} "
        f"| {s['ratio']:.0f} | {s['ratio_without_top']:.0f} |"
    )

out = Path("results/act_stats") / f"{Path(args.model).name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(
        {
            "model": args.model,
            "window": args.window,
            "seqlen": args.seqlen,
            "stats": [{"block": b, "input": INPUTS[k], **v} for (b, k), v in sorted(stats.items())],
        },
        indent=2,
    )
)
print("\nSaved to", out)
