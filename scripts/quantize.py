"""Quantize a model with llm-compressor and record what the quantization cost.

Methods: RTN, GPTQ, AWQ (group A, W4A16) and SmoothQuant (group B, W8A8).

Usage (from ~/Q_experiments):
  python scripts/quantize.py --method rtn  --model meta-llama/Llama-3.2-1B
  python scripts/quantize.py --method gptq --model meta-llama/Llama-3.2-1B
  python scripts/quantize.py --method awq  --model meta-llama/Llama-3.2-1B
  python scripts/quantize.py --method rtn  --scheme W8A8 --model meta-llama/Llama-3.2-1B
  python scripts/quantize.py --method sq   --scheme W8A8 --model meta-llama/Llama-3.2-1B
  python scripts/quantize.py --method sq   --scheme W8A8 --alpha 0.8 --model meta-llama/Llama-3.2-1B

Outputs:
  models/<run name>/             compressed checkpoint; recipe.yaml inside lists every setting used
  results/quant/<run name>.json  quantization time, peak GPU/CPU memory, size on disk
  run name = <model>-<METHOD>-<scheme>[-alpha<a>][-seed<n>], e.g. Llama-3.2-1B-SQ-W8A8-seed1
"""
import argparse
import json
import resource
import time
from pathlib import Path

import llmcompressor
import torch
from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

from calib_data import get_calibration_data

DEFAULT_ALPHA = 0.5  # SmoothQuant paper's default smoothing strength (also llm-compressor's default)

# Formats that are not compressed-tensors presets.
CUSTOM_SCHEMES = {
    # Same INT8 weights as the W8A8 preset, but ONE fixed activation scale per layer, measured on the
    # calibration data, instead of a new scale for every token at run time. Cheaper at inference, but a
    # single outlier channel then squeezes every token. This is the setting SmoothQuant was designed for.
    "W8A8_STATIC": QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="channel"),
        input_activations=QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor", dynamic=False),
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Hugging Face model ID")
parser.add_argument("--method", required=True, choices=["rtn", "gptq", "awq", "sq"])
parser.add_argument(
    "--scheme",
    default="W4A16_ASYM",
    help="W4A16_ASYM = 4-bit weights with a zero-point, one scale per group of 128 weights, BF16 activations. "
    "W8A8 = INT8 weights (one scale per output channel) and INT8 activations (one scale per token, computed "
    "at run time). W8A8_STATIC = as W8A8, but one fixed activation scale per layer from calibration",
)
parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="SmoothQuant smoothing strength (0-1)")
parser.add_argument("--n-samples", type=int, default=128)
parser.add_argument("--seqlen", type=int, default=2048)
parser.add_argument("--seed", type=int, default=0, help="which random calibration samples to use (0 = our standard set)")
args = parser.parse_args()


def scheme_kwargs(scheme):
    """Presets are passed by name; our custom formats as an explicit config group."""
    if scheme in CUSTOM_SCHEMES:
        return {"config_groups": {"group_0": CUSTOM_SCHEMES[scheme]}}
    return {"scheme": scheme}


def build_recipe(method, scheme, alpha):
    quantize = dict(targets="Linear", ignore=["lm_head"], **scheme_kwargs(scheme))
    if method == "rtn":
        # Round-to-nearest: each weight is simply rounded to the closest level.
        # No error correction: the reference that shows what the other methods add.
        return QuantizationModifier(**quantize)
    if method == "gptq":
        # GPTQ rounds each layer's weights one column at a time. After each column it adjusts the
        # columns not yet rounded to cancel the rounding error, using statistics of the calibration
        # inputs to know which directions matter.
        return GPTQModifier(**quantize)
    if method == "awq":
        # AWQ first scales up the weight channels that receive large activations (the important ones),
        # so rounding hurts them less. Then plain round-to-nearest quantization does the rounding.
        return [AWQModifier(duo_scaling="both"), QuantizationModifier(**quantize)]
    # SmoothQuant divides each activation channel by s and multiplies the matching weight row by s, so the
    # result is unchanged: s_j = max|X_j|^alpha / max|W_j|^(1-alpha). Outlier channels shrink, which makes
    # activations easier to quantize, and the weights absorb the difficulty. Then round-to-nearest, as in
    # the paper.
    return [SmoothQuantModifier(smoothing_strength=alpha), QuantizationModifier(**quantize)]


model_name = Path(args.model).name
run_name = f"{model_name}-{args.method.upper()}-{args.scheme}"
if args.method == "sq" and args.alpha != DEFAULT_ALPHA:
    run_name += f"-alpha{args.alpha}"
if args.seed:
    run_name += f"-seed{args.seed}"
save_dir = Path("models") / run_name

tokenizer = AutoTokenizer.from_pretrained(args.model)
# Loaded on the CPU on purpose: llm-compressor moves one layer at a time to the GPU.
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
calib = get_calibration_data(tokenizer, model_name, args.n_samples, args.seqlen, seed=args.seed)

# Quantization cost: time and memory of the algorithm itself (not loading or saving the model).
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
oneshot(
    model=model,
    dataset=calib,
    recipe=build_recipe(args.method, args.scheme, args.alpha),
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

model.save_pretrained(save_dir, save_compressed=True)  # writes the real low-bit format
tokenizer.save_pretrained(save_dir)
size_gib = sum(f.stat().st_size for f in save_dir.glob("*.safetensors")) / 2**30

result = {
    "model": args.model,
    "method": args.method,
    "scheme": args.scheme,
    "smoothing_strength": args.alpha if args.method == "sq" else None,
    "calibration": f"C4, {args.n_samples} x {args.seqlen} tokens, seed {args.seed}",
    "seed": args.seed,
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
