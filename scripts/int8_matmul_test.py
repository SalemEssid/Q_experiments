"""Why is W8A8 prefill only 1.3-1.4x faster than BF16, when INT8 tensor cores have 2x the BF16 peak?

Times the pieces of one W8A8 linear layer exactly as vLLM 0.32 runs it (CutlassInt8ScaledMMLinearKernel):
  1. ops.scaled_int8_quant(x): BF16 activations -> INT8 plus one scale per token. A separate kernel
     (the norm/activation fusion passes are off in our vLLM config).
  2. ops.cutlass_scaled_mm(x_q, w_q, x_s, w_s): INT8 x INT8 matmul, scales applied at the end, BF16 output.
and compares them with the BF16 layer (F.linear, i.e. cuBLAS, which vLLM uses for unquantized layers).

Shapes: the Llama-3.2 linear layers as vLLM fuses them (q/k/v -> qkv_proj, gate/up -> gate_up_proj), for
  M = 8192 tokens per step (prefill: vLLM's max_num_batched_tokens here) and
  M = 256 tokens per step (one generation step with 256 running requests, as in the throughput test).
Then it adds up all layers and splits the measured end-to-end prefill time (results/speed) into
matmuls, activation quantization and everything else (attention, norms, ...).

Usage (from ~/Q_experiments, vllm venv active, nothing else running on the GPU; about 2 minutes):
  python scripts/int8_matmul_test.py 2>&1 | tee logs/int8_matmul_test.log
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops

# (N output features, K input features) of each linear layer in one block
MODELS = {
    "Llama-3.2-1B": {
        "layers": 16,
        "linears": {"qkv_proj": (3072, 2048), "o_proj": (2048, 2048), "gate_up_proj": (16384, 2048), "down_proj": (2048, 8192)},
        "lm_head": (128256, 2048),
    },
    "Llama-3.2-3B": {
        "layers": 28,
        "linears": {"qkv_proj": (5120, 3072), "o_proj": (3072, 3072), "gate_up_proj": (16384, 3072), "down_proj": (3072, 8192)},
        "lm_head": (128256, 3072),
    },
}


def time_us(fn, iters=30, repeats=5):
    """Median time of one call in microseconds (CUDA events, after a warm-up)."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / iters)
    return statistics.median(times)


def int8_weight(n, k):
    """Random INT8 weight stored as vLLM stores it: [K, N] column-major (weight.t()), one scale per output channel."""
    w_q = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8).t()
    w_s = torch.rand(n, 1, device="cuda", dtype=torch.float32) * 1e-3 + 1e-4
    return w_q, w_s


def check_kernels():
    """The INT8 path must compute the same thing as a plain float matmul of the dequantized values."""
    m, n, k = 64, 256, 512
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w_q, w_s = int8_weight(n, k)
    x_q, x_s, _ = ops.scaled_int8_quant(x)
    out = ops.cutlass_scaled_mm(x_q, w_q, x_s, w_s, torch.bfloat16)
    ref = (x_q.float() @ w_q.float()) * x_s * w_s.t()
    mm_err = ((out.float() - ref).norm() / ref.norm()).item()
    q_err = ((x_q.float() * x_s - x.float()).abs().max() / x.float().abs().max()).item()
    print(f"Kernel check: INT8 matmul vs float reference, relative error {mm_err:.1e} (BF16 output rounding: about 1e-3)")
    print(f"              per-token quantization, max error / max value {q_err:.1e} (INT8 rounding: at most 1/254 = 3.9e-3)")
    assert mm_err < 1e-2 and q_err < 5e-3, "the INT8 kernels do not compute what this test assumes"


def square_probe(n=8192):
    """Peak speed on a large square matmul: BF16 (cuBLAS), INT8 with scales (vLLM's CUTLASS), raw INT8 (cuBLAS)."""
    flops = 2 * n**3
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    a_q, a_s, _ = ops.scaled_int8_quant(a)
    w_q, w_s = int8_weight(n, n)
    result = {
        "bf16_tflops": flops / time_us(lambda: a @ b, iters=10) / 1e6,
        "int8_cutlass_tops": flops / time_us(lambda: ops.cutlass_scaled_mm(a_q, w_q, a_s, w_s, torch.bfloat16), iters=10) / 1e6,
    }
    try:
        result["int8_cublas_raw_tops"] = flops / time_us(lambda: torch._int_mm(a_q, w_q), iters=10) / 1e6
    except Exception as e:  # not every build or layout supports torch._int_mm
        result["int8_cublas_raw_tops"] = None
        print("  torch._int_mm not available:", str(e).splitlines()[0])
    del a, b, a_q, w_q
    torch.cuda.empty_cache()
    return result


def time_linear(m, n, k):
    """Times of every piece of one linear layer with M tokens (microseconds)."""
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    w_q, w_s = int8_weight(n, k)
    static_scale = torch.tensor(0.05, device="cuda", dtype=torch.float32)  # one scale per layer (W8A8_STATIC)
    x_q, x_s, _ = ops.scaled_int8_quant(x)
    t = {
        "bf16": time_us(lambda: F.linear(x, w)),
        "quant_dynamic": time_us(lambda: ops.scaled_int8_quant(x)),
        "quant_static": time_us(lambda: ops.scaled_int8_quant(x, static_scale)),
        "int8_matmul": time_us(lambda: ops.cutlass_scaled_mm(x_q, w_q, x_s, w_s, torch.bfloat16)),
        "int8_matmul_static": time_us(lambda: ops.cutlass_scaled_mm(x_q, w_q, static_scale, w_s, torch.bfloat16)),
    }
    del x, w, w_q, x_q
    torch.cuda.empty_cache()
    return t


def load_prefill(name):
    path = Path("results/speed") / f"{name}.json"
    return json.loads(path.read_text())["prefill"]["tokens_per_s"] if path.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="8192,256", help="tokens per step to test (comma-separated)")
    parser.add_argument("--out", default="results/int8_test/int8_matmul_test.json")
    args = parser.parse_args()
    token_counts = [int(m) for m in args.tokens.split(",")]

    print("GPU:", torch.cuda.get_device_name(0))
    check_kernels()
    probe = square_probe()
    ratio = probe["int8_cutlass_tops"] / probe["bf16_tflops"]
    raw = f"{probe['int8_cublas_raw_tops']:.1f}" if probe["int8_cublas_raw_tops"] else "n/a"
    print(f"\nSquare 8192 x 8192 matmul: BF16 {probe['bf16_tflops']:.1f} TFLOPS | INT8 (vLLM CUTLASS) "
          f"{probe['int8_cutlass_tops']:.1f} TOPS | INT8 raw cuBLAS {raw} TOPS | INT8 / BF16 = {ratio:.2f}x")

    results = {"gpu": torch.cuda.get_device_name(0), "probe": probe, "models": {}}
    for model, spec in MODELS.items():
        results["models"][model] = {}
        for m in token_counts:
            print(f"\n{model}, {m} tokens per step (times in microseconds, for one layer)")
            print("| Layer (N x K) | BF16 | INT8 matmul | Quantize activations | INT8 total | Matmul speedup | Total speedup | BF16 TFLOPS | INT8 TOPS |")
            print("|---|---|---|---|---|---|---|---|---|")
            rows = {}
            for layer, (n, k) in spec["linears"].items():
                t = time_linear(m, n, k)
                rows[layer] = t
                flops = 2 * m * n * k
                total = t["quant_dynamic"] + t["int8_matmul"]
                print(f"| {layer} ({n} x {k}) | {t['bf16']:.0f} | {t['int8_matmul']:.0f} | {t['quant_dynamic']:.0f} | {total:.0f} "
                      f"| {t['bf16'] / t['int8_matmul']:.2f}x | {t['bf16'] / total:.2f}x "
                      f"| {flops / t['bf16'] / 1e6:.0f} | {flops / t['int8_matmul'] / 1e6:.0f} |")

            # whole model, one step of m tokens: every block has these four layers
            layers = spec["layers"]
            bf16 = layers * sum(t["bf16"] for t in rows.values())
            matmul = layers * sum(t["int8_matmul"] for t in rows.values())
            quant = layers * sum(t["quant_dynamic"] for t in rows.values())
            static = layers * sum(t["quant_static"] + t["int8_matmul_static"] for t in rows.values())
            summary = {"bf16_us": bf16, "int8_matmul_us": matmul, "quant_us": quant, "int8_static_total_us": static}
            print(f"All {layers} blocks: BF16 {bf16 / 1000:.2f} ms | INT8 matmuls {matmul / 1000:.2f} ms + quantization "
                  f"{quant / 1000:.2f} ms = {(matmul + quant) / 1000:.2f} ms ({bf16 / (matmul + quant):.2f}x) | "
                  f"static scales {static / 1000:.2f} ms ({bf16 / static:.2f}x) | quantization = "
                  f"{quant / (matmul + quant):.0%} of the INT8 layer time")

            if m <= 512:  # generation step: lm_head runs for every token and stays BF16 in both formats
                n, k = spec["lm_head"]
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
                lm = time_us(lambda: F.linear(x, w))
                del x, w
                torch.cuda.empty_cache()
                summary["lm_head_bf16_us"] = lm
                print(f"lm_head (BF16 in both formats): {lm / 1000:.2f} ms per step = {lm / (bf16 + lm):.0%} of the BF16 matmul time, "
                      f"{lm / (matmul + quant + lm):.0%} of the W8A8 matmul time")

            measured_bf16, measured_w8a8 = load_prefill(model), load_prefill(f"{model}-RTN-W8A8")
            if m >= 2048 and measured_bf16 and measured_w8a8:
                # per token, in microseconds; "other" = everything that is not a linear layer, the same in both formats
                e2e_bf16, e2e_w8a8 = 1e6 / measured_bf16, 1e6 / measured_w8a8
                other = e2e_bf16 - bf16 / m
                other_w8a8 = e2e_w8a8 - (matmul + quant) / m
                predicted = other + (matmul + quant) / m
                print(f"\nPrefill, measured in vLLM: BF16 {e2e_bf16:.2f} us/token, W8A8 {e2e_w8a8:.2f} us/token ({e2e_bf16 / e2e_w8a8:.2f}x)")
                print(f"  BF16 : linear layers {bf16 / m:.2f} us/token ({bf16 / m / e2e_bf16:.0%}), everything else {other:.2f} us/token ({other / e2e_bf16:.0%})")
                print(f"  W8A8 : INT8 matmuls {matmul / m:.2f} ({matmul / m / e2e_w8a8:.0%}), quantization {quant / m:.2f} ({quant / m / e2e_w8a8:.0%}), "
                      f"everything else {other_w8a8:.2f} us/token ({other_w8a8 / e2e_w8a8:.0%})")
                print(f"  Consistency: 'everything else' from BF16 ({other:.2f}) vs from W8A8 ({other_w8a8:.2f}) should be close")
                print(f"  Predicted W8A8 speedup from the parts: {e2e_bf16 / predicted:.2f}x (measured {e2e_bf16 / e2e_w8a8:.2f}x)")
                print(f"  If quantization were free (fused into the previous kernel): {e2e_bf16 / (other + matmul / m):.2f}x")
                print(f"  If INT8 matmuls were exactly 2x BF16 and quantization free: {e2e_bf16 / (other + bf16 / m / 2):.2f}x")
                summary.update({"measured_prefill_bf16_us_per_token": e2e_bf16, "measured_prefill_w8a8_us_per_token": e2e_w8a8,
                                "other_us_per_token_from_bf16": other, "other_us_per_token_from_w8a8": other_w8a8})
            results["models"][model][str(m)] = {"layers": rows, "summary": summary}

    probe_after = square_probe()
    print(f"\nSquare matmul again at the end: BF16 {probe_after['bf16_tflops']:.1f} TFLOPS, INT8 {probe_after['int8_cutlass_tops']:.1f} TOPS")
    results["probe_after"] = probe_after
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("Saved to", out)


if __name__ == "__main__":
    main()
