"""Collect every result JSON into Markdown tables, ready to paste into the report.

Table 1: one row per run. Table 2: mean ± std over calibration seeds (only when a method has several seeds).
Table 3: vLLM speed and memory (only when results/speed exists).
Usage (from ~/Q_experiments):  python scripts/summarize.py
"""
import json
import re
import statistics
from pathlib import Path

METHOD_ORDER = {"BF16": 0, "RTN": 1, "GPTQ": 2, "AWQ": 3}


def load(folder):
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(folder).glob("*.json"))}


ppl = load("results/ppl")
quant = load("results/quant")

rows = []
seed_groups = {}  # (model, method order, method, scheme) -> PPL of each seed
for run, r in ppl.items():
    q = quant.get(run)  # None for the BF16 baselines
    base_name = Path(q["model"]).name if q else run
    base_ppl = ppl.get(base_name, {}).get("ppl")
    rtn_ppl = ppl.get(f"{base_name}-RTN-{q['scheme']}", {}).get("ppl") if q else None
    method = q["method"].upper() if q else "BF16"
    seed = q.get("seed", 0) if q else 0
    label = method + (f" (seed {seed})" if seed else "")
    order = METHOD_ORDER.get(method, 9)

    delta = f"{r['ppl'] - base_ppl:+.3f}" if q and base_ppl else "–"
    # Share of RTN's perplexity loss that the method wins back (0% = RTN, 100% = BF16).
    recovered = (
        f"{(rtn_ppl - r['ppl']) / (rtn_ppl - base_ppl):.0%}"
        if q and rtn_ppl and base_ppl and method != "RTN"
        else "–"
    )
    rows.append(
        (
            (base_name, order, seed),
            f"| {base_name} | {label} | {q['scheme'] if q else 'BF16'} | {r['ppl']:.3f} | {delta} | {recovered} "
            f"| {q['size_on_disk_gib'] if q else '–'} | {q['quant_seconds'] if q else '–'} "
            f"| {q['peak_gpu_gib'] if q else '–'} |",
        )
    )
    if q:
        seed_groups.setdefault((base_name, order, method, q["scheme"]), []).append(r["ppl"])

print("| Model | Method | Scheme | WikiText-2 PPL | Δ vs BF16 | RTN loss recovered | Size (GiB) | Quant time (s) | Quant peak GPU (GiB) |")
print("|---|---|---|---|---|---|---|---|---|")
for _, line in sorted(rows):
    print(line)

multi_seed = [(k, v) for k, v in sorted(seed_groups.items()) if len(v) > 1]
if multi_seed:
    print("\nCalibration-seed study: WikiText-2 PPL, mean ± sample std over seeds")
    print("| Model | Method | Scheme | Seeds | PPL mean ± std | Min | Max | RTN loss recovered (mean) |")
    print("|---|---|---|---|---|---|---|---|")
    for (model, _, method, scheme), values in multi_seed:
        mean = statistics.mean(values)
        base_ppl = ppl[model]["ppl"]
        rtn_ppl = ppl.get(f"{model}-RTN-{scheme}", {}).get("ppl")
        recovered = f"{(rtn_ppl - mean) / (rtn_ppl - base_ppl):.0%}" if rtn_ppl else "–"
        print(
            f"| {model} | {method} | {scheme} | {len(values)} | {mean:.3f} ± {statistics.stdev(values):.3f} "
            f"| {min(values):.3f} | {max(values):.3f} | {recovered} |"
        )


def split_name(name):
    """'Llama-3.2-1B-GPTQ-W4A16_ASYM' -> ('Llama-3.2-1B', 'GPTQ'); 'Llama-3.2-1B' -> ('Llama-3.2-1B', 'BF16')."""
    for m in ("RTN", "GPTQ", "AWQ"):
        if f"-{m}-" in name:
            return name[: name.index(f"-{m}-")], m
    return name, "BF16"


speed = load("results/speed")
if speed:
    print("\nSpeed and memory (vLLM, A40-24Q, medians). Speedups are relative to the BF16 model of the same size.")
    print(
        "| Model | Method | Weights in GPU (GiB) | KV cache (k tokens) | TTFT 128 tok (ms) | TPOT bs1 (ms) "
        "| Decode tok/s bs1 | Prefill tok/s | Throughput out tok/s | GPU TFLOPS before / after |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for name, s in speed.items():
        base, method = split_name(name)
        rows.append(((base, METHOD_ORDER.get(method, 9)), name, base, method, s))
    for _, name, base, method, s in sorted(rows):
        log = Path("logs") / f"bench_{name}.log"
        match = re.search(r"Model loading took ([\d.]+) GiB", log.read_text(errors="ignore")) if log.exists() else None
        weights = match.group(1) if match else "?"
        kv = f"{s['kv_cache_tokens'] / 1000:.0f}" if s.get("kv_cache_tokens") else "?"
        ref = speed.get(base)
        lat, pre, thr = s["latency_bs1"], s["prefill"], s["throughput"]

        def vs_bf16(value, ref_value):
            return f"{value} ({value / ref_value:.2f}×)" if ref and method != "BF16" else f"{value}"

        print(
            f"| {base} | {method} | {weights} | {kv} | {lat['ttft_ms']} | {lat['tpot_ms']} "
            f"| {vs_bf16(lat['decode_tokens_per_s'], ref['latency_bs1']['decode_tokens_per_s'] if ref else 1)} "
            f"| {vs_bf16(pre['tokens_per_s'], ref['prefill']['tokens_per_s'] if ref else 1)} "
            f"| {vs_bf16(thr['output_tokens_per_s'], ref['throughput']['output_tokens_per_s'] if ref else 1)} "
            f"| {s['gpu_tflops_before']} / {s['gpu_tflops_after']} |"
        )
