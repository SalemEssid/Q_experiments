"""Quantize a model with llm-compressor (RTN, GPTQ or AWQ) and record what the quantization cost.

Usage (from ~/Q_experiments):
  python scripts/quantize.py --method rtn  --model meta-llama/Llama-3.2-1B
  python scripts/quantize.py --method gptq --model meta-llama/Llama-3.2-1B
  python scripts/quantize.py --method awq  --model meta-llama/Llama-3.2-1B

Outputs:
  models/<model>-<METHOD>-<scheme>/             compressed checkpoint; recipe.yaml inside lists every setting used
  results/quant/<model>-<METHOD>-<scheme>.json  quantization time, peak GPU/CPU memory, size on disk
"""
import argparse
import json
import resource
import time
from pathlib import Path

import llmcompressor
import torch
from compressed_tensors.offload import dispatch_model
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

from calib_data import get_calibration_data

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Hugging Face model ID")
parser.add_argument("--method", required=True, choices=["rtn", "gptq", "awq"])
parser.add_argument(
    "--scheme",
    default="W4A16_ASYM",
    help="compressed-tensors preset. W4A16_ASYM = 4-bit weights with a zero-point, "
    "one scale per group of 128 weights, activations stay BF16",
)
parser.add_argument("--n-samples", type=int, default=128)
parser.add_argument("--seqlen", type=int, default=2048)
args = parser.parse_args()


def build_recipe(method, scheme):
    if method == "rtn":
        # Round-to-nearest: each weight is simply rounded to the closest 4-bit level.
        # No calibration and no error correction: the reference that shows what GPTQ and AWQ add.
        return QuantizationModifier(targets="Linear", scheme=scheme, ignore=["lm_head"])
    if method == "gptq":
        # GPTQ rounds each layer's weights one column at a time. After each column it adjusts the
        # columns not yet rounded to cancel the rounding error, using statistics of the calibration
        # inputs to know which directions matter.
        return GPTQModifier(targets="Linear", scheme=scheme, ignore=["lm_head"])
    # AWQ first scales up the weight channels that receive large activations (the important ones),
    # so rounding hurts them less. Then plain round-to-nearest quantization does the rounding.
    return [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(targets="Linear", scheme=scheme, ignore=["lm_head"]),
    ]


model_name = Path(args.model).name
run_name = f"{model_name}-{args.method.upper()}-{args.scheme}"
save_dir = Path("models") / run_name

tokenizer = AutoTokenizer.from_pretrained(args.model)
# Loaded on the CPU on purpose: llm-compressor moves one layer at a time to the GPU.
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
calib = get_calibration_data(tokenizer, model_name, args.n_samples, args.seqlen)

# Quantization cost: time and memory of the algorithm itself (not loading or saving the model).
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
oneshot(
    model=model,
    dataset=calib,
    recipe=build_recipe(args.method, args.scheme),
    max_seq_length=args.seqlen,
    num_calibration_samples=args.n_samples,
    shuffle_calibration_samples=False,
)
quant_seconds = time.perf_counter() - start
peak_gpu_gib = torch.cuda.max_memory_allocated() / 2**30

# Sanity check: a broken quantization usually produces gibberish here.
dispatch_model(model)
prompt = tokenizer("The capital of France is", return_tensors="pt")
prompt = {k: v.to(model.device) for k, v in prompt.items()}
sample = tokenizer.decode(model.generate(**prompt, max_new_tokens=20, do_sample=False)[0])
print("\nSample generation:", sample, "\n")

model.save_pretrained(save_dir, save_compressed=True)  # writes the real packed low-bit format
tokenizer.save_pretrained(save_dir)
size_gib = sum(f.stat().st_size for f in save_dir.glob("*.safetensors")) / 2**30

result = {
    "model": args.model,
    "method": args.method,
    "scheme": args.scheme,
    "calibration": f"C4, {args.n_samples} x {args.seqlen} tokens, seed 0",
    "quant_seconds": round(quant_seconds, 1),
    "peak_gpu_gib": round(peak_gpu_gib, 2),
    "peak_cpu_ram_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
    "size_on_disk_gib": round(size_gib, 3),
    "sample_generation": sample,
    "llmcompressor": llmcompressor.__version__,
    "torch": torch.__version__,
}
out = Path("results/quant") / f"{run_name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("Checkpoint saved to", save_dir)
print("Results saved to", out)
