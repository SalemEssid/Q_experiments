"""Speed benchmark with vLLM: single-request latency, prefill speed and throughput under load.

Usage (from ~/Q_experiments, with the vllm venv active):
  python scripts/bench_vllm.py --model models/Llama-3.2-1B-GPTQ-W4A16_ASYM 2>&1 | tee logs/bench_Llama-3.2-1B-GPTQ-W4A16_ASYM.log
scripts/run_speed_bench.sh runs it for every model. Keep the log: summarize.py reads the weight memory from it.

Workloads. Prompts are random tokens of a fixed length, identical for every model. Decoding is greedy
with ignore_eos, so every request produces exactly the requested number of tokens. Times are medians.
  1. Latency, one request (memory-bound: favours small weights)
       TTFT = time to first token for a 128-token prompt
       TPOT = time per output token while generating 256 tokens
  2. Prefill: 32 requests x 1024 prompt tokens -> 1 token (compute-bound: favours fast math)
  3. Throughput: 256 requests x 512 prompt tokens -> 256 new tokens, all submitted at once (serving under load)
Prefix caching is off, so repeated prompts are really recomputed.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

# Same workaround as vllm_check.py: the system nvcc 12.0 is too old for FlashInfer's sampler.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
import vllm
from vllm import LLM, SamplingParams

GPU_MEMORY_UTILIZATION = 0.85  # same memory budget for every model
MAX_MODEL_LEN = 2048


def gpu_tflops(n=8192, iters=20):
    """BF16 matmul probe: about 110 TFLOPS when the shared physical GPU is idle."""
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    tflops = 2 * n**3 * iters / (time.perf_counter() - start) / 1e12
    del a, b
    torch.cuda.empty_cache()
    return round(tflops, 1)


def random_prompts(n, length, seed):
    """n prompts of exactly `length` random token IDs."""
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, 100000, (n, length), generator=gen).tolist()
    return [{"prompt_token_ids": x} for x in ids]


def median_time(llm, prompts, max_tokens, warmup, repeats):
    params = SamplingParams(max_tokens=max_tokens, temperature=0, ignore_eos=True)
    for _ in range(warmup):  # the first runs are slower (compilation, caches)
        llm.generate(prompts, params, use_tqdm=False)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        llm.generate(prompts, params, use_tqdm=False)
        times.append(time.perf_counter() - start)
    return statistics.median(times), [round(t, 4) for t in times]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", default="results/speed")
    args = parser.parse_args()

    tflops_before = gpu_tflops()
    print(f"GPU check before: {tflops_before} TFLOPS")

    llm = LLM(
        model=args.model,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=False,
        seed=0,
    )
    try:  # how many tokens fit in the KV cache with this memory budget
        cache = llm.llm_engine.vllm_config.cache_config
        kv_cache_tokens = cache.num_gpu_blocks * cache.block_size
    except Exception:
        kv_cache_tokens = None

    print("1/3 latency (1 request)")
    one = random_prompts(1, 128, seed=0)
    ttft, ttft_runs = median_time(llm, one, max_tokens=1, warmup=3, repeats=10)
    e2e, e2e_runs = median_time(llm, one, max_tokens=256, warmup=3, repeats=10)
    tpot = (e2e - ttft) / 255  # the 255 tokens generated after the first one

    print("2/3 prefill (32 x 1024 tokens)")
    prefill = random_prompts(32, 1024, seed=1)
    t_prefill, prefill_runs = median_time(llm, prefill, max_tokens=1, warmup=1, repeats=5)

    print("3/3 throughput (256 requests x 512 -> 256 tokens)")
    many = random_prompts(256, 512, seed=2)
    t_thr, thr_runs = median_time(llm, many, max_tokens=256, warmup=1, repeats=3)

    tflops_after = gpu_tflops()

    result = {
        "model": args.model,
        "gpu_tflops_before": tflops_before,
        "gpu_tflops_after": tflops_after,
        "kv_cache_tokens": kv_cache_tokens,
        "latency_bs1": {
            "prompt_tokens": 128,
            "output_tokens": 256,
            "ttft_ms": round(ttft * 1000, 2),
            "tpot_ms": round(tpot * 1000, 3),
            "decode_tokens_per_s": round(1 / tpot, 1),
            "e2e_ms": round(e2e * 1000, 1),
            "ttft_runs_s": ttft_runs,
            "e2e_runs_s": e2e_runs,
        },
        "prefill": {
            "requests": 32,
            "prompt_tokens": 1024,
            "tokens_per_s": round(32 * 1024 / t_prefill),
            "runs_s": prefill_runs,
        },
        "throughput": {
            "requests": 256,
            "prompt_tokens": 512,
            "output_tokens": 256,
            "output_tokens_per_s": round(256 * 256 / t_thr),
            "total_tokens_per_s": round(256 * (512 + 256) / t_thr),
            "runs_s": thr_runs,
        },
        "settings": {
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_model_len": MAX_MODEL_LEN,
            "prefix_caching": False,
            "flashinfer_sampler": False,
        },
        "vllm": vllm.__version__,
        "torch": torch.__version__,
    }
    out = Path(args.out_dir) / f"{Path(args.model).name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "settings"}, indent=2))
    print("Saved to", out)


# vLLM starts helper processes; this guard stops them from re-running the script.
if __name__ == "__main__":
    main()
