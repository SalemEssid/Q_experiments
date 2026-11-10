"""Collect every result JSON into Markdown tables, ready to paste into the report.

Table 1: one row per run. Table 2: mean ± std over calibration seeds (only when a configuration has several seeds).
Table 3: vLLM speed and memory (only when results/speed exists).
Tables 4-5: zero-shot accuracy per run, and its average over calibration seeds (only when results/zeroshot exists).
Rows are grouped by model, then by format (BF16, W4A16, W8A8, ...), then by method.
Usage (from ~/Q_experiments):  python scripts/summarize.py
"""
import json
import re
import statistics
from pathlib import Path

METHOD_ORDER = {"BF16": 0, "RTN": 1, "GPTQ": 2, "AWQ": 3, "SQ": 4}
METHOD_LABELS = {"SQ": "SmoothQuant"}


def load(folder):
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(folder).glob("*.json"))}


def parse_name(name):
    """Split a run name into (model, method, scheme, variant, seed).

    'Llama-3.2-1B-SQ-W8A8-alpha0.8-seed1' -> ('Llama-3.2-1B', 'SQ', 'W8A8', 'alpha0.8', 1)
    'Llama-3.2-1B-GPTQ-W4A16_ASYM'        -> ('Llama-3.2-1B', 'GPTQ', 'W4A16_ASYM', '', 0)
    'Llama-3.2-1B'                        -> ('Llama-3.2-1B', 'BF16', 'BF16', '', 0)
    """
    seed = 0
    match = re.search(r"-seed(\d+)$", name)
    if match:
        seed, name = int(match.group(1)), name[: match.start()]
    for method in METHOD_ORDER:
        if f"-{method}-" in name:
            base, rest = name.split(f"-{method}-", 1)
            scheme, _, variant = rest.partition("-")
            return base, method, scheme, variant, seed
    return name, "BF16", "BF16", "", seed


def sort_key(name):
    base, method, scheme, variant, seed = parse_name(name)
    return (base, scheme, METHOD_ORDER[method], variant, seed)  # "BF16" sorts before the "W..." formats


def config_key(name):
    """Everything except the seed: runs with the same key are repeats with different calibration samples."""
    return sort_key(name)[:4]


def label(name, with_seed=True):
    _, method, _, variant, seed = parse_name(name)
    text = METHOD_LABELS.get(method, method)
    if variant.startswith("alpha"):
        text += f" α={variant[len('alpha'):]}"
    if with_seed and seed:
        text += f" (seed {seed})"
    return text


ppl = load("results/ppl")
quant = load("results/quant")


def rtn_reference(name):
    base, _, scheme, _, _ = parse_name(name)
    return ppl.get(f"{base}-RTN-{scheme}", {}).get("ppl")


def recovered(value, rtn_value, base_value):
    """Share of RTN's perplexity loss that a method wins back (0% = RTN, 100% = BF16)."""
    if rtn_value is None or base_value is None or rtn_value <= base_value:
        return "–"
    return f"{(rtn_value - value) / (rtn_value - base_value):.0%}"


print(
    "| Model | Method | Scheme | WikiText-2 PPL | Δ vs BF16 | RTN loss recovered | Size (GiB) "
    "| Quant time (s) | Quant peak GPU (GiB) |"
)
print("|---|---|---|---|---|---|---|---|---|")
seed_groups = {}  # config key -> [(name, PPL)]
for name in sorted(ppl, key=sort_key):
    base, method, scheme, _, _ = parse_name(name)
    value = ppl[name]["ppl"]
    q = quant.get(name)
    base_ppl = ppl.get(base, {}).get("ppl")
    is_quantized = method != "BF16"
    delta = f"{value - base_ppl:+.3f}" if is_quantized and base_ppl else "–"
    rec = recovered(value, rtn_reference(name), base_ppl) if method not in ("BF16", "RTN") else "–"
    print(
        f"| {base} | {label(name)} | {scheme} | {value:.3f} | {delta} | {rec} "
        f"| {q['size_on_disk_gib'] if q else '–'} | {q['quant_seconds'] if q else '–'} "
        f"| {q['peak_gpu_gib'] if q else '–'} |"
    )
    if is_quantized:
        seed_groups.setdefault(config_key(name), []).append((name, value))

multi_seed = [runs for _, runs in sorted(seed_groups.items()) if len(runs) > 1]
if multi_seed:
    print("\nCalibration-seed study: WikiText-2 PPL, mean ± sample std over seeds")
    print("| Model | Method | Scheme | Seeds | PPL mean ± std | Min | Max | RTN loss recovered (mean) |")
    print("|---|---|---|---|---|---|---|---|")
    for runs in multi_seed:
        first = runs[0][0]
        base, _, scheme, _, _ = parse_name(first)
        values = [v for _, v in runs]
        mean = statistics.mean(values)
        rec = recovered(mean, rtn_reference(first), ppl.get(base, {}).get("ppl"))
        print(
            f"| {base} | {label(first, with_seed=False)} | {scheme} | {len(values)} "
            f"| {mean:.3f} ± {statistics.stdev(values):.3f} | {min(values):.3f} | {max(values):.3f} | {rec} |"
        )


speed = load("results/speed")
if speed:
    print("\nSpeed and memory (vLLM, A40-24Q, medians). Speedups are relative to the BF16 model of the same size.")
    print(
        "| Model | Method | Scheme | Weights in GPU (GiB) | KV cache (k tokens) | TTFT 128 tok (ms) | TPOT bs1 (ms) "
        "| Decode tok/s bs1 | Prefill tok/s | Throughput out tok/s | GPU TFLOPS before / after |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in sorted(speed, key=sort_key):
        s = speed[name]
        base, method, scheme, _, _ = parse_name(name)
        log = Path("logs") / f"bench_{name}.log"
        match = re.search(r"Model loading took ([\d.]+) GiB", log.read_text(errors="ignore")) if log.exists() else None
        weights = match.group(1) if match else "?"
        kv = f"{s['kv_cache_tokens'] / 1000:.0f}" if s.get("kv_cache_tokens") else "?"
        ref = speed.get(base) if method != "BF16" else None
        lat, pre, thr = s["latency_bs1"], s["prefill"], s["throughput"]

        def vs_bf16(value, ref_value):
            return f"{value} ({value / ref_value:.2f}×)" if ref else f"{value}"

        print(
            f"| {base} | {label(name)} | {scheme} | {weights} | {kv} | {lat['ttft_ms']} | {lat['tpot_ms']} "
            f"| {vs_bf16(lat['decode_tokens_per_s'], ref['latency_bs1']['decode_tokens_per_s'] if ref else 1)} "
            f"| {vs_bf16(pre['tokens_per_s'], ref['prefill']['tokens_per_s'] if ref else 1)} "
            f"| {vs_bf16(thr['output_tokens_per_s'], ref['throughput']['output_tokens_per_s'] if ref else 1)} "
            f"| {s['gpu_tflops_before']} / {s['gpu_tflops_after']} |"
        )


TASK_LABELS = {
    "arc_easy": "ARC-e",
    "arc_challenge": "ARC-c",
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "winogrande": "WinoGrande",
    "boolq": "BoolQ",
}

zeroshot = load("results/zeroshot")
if zeroshot:
    print("\nZero-shot accuracy (%, lm-eval, 0-shot; acc_norm where the task has it, else acc)")
    print("| Model | Method | Scheme | " + " | ".join(TASK_LABELS.values()) + " | Avg |")
    print("|---|---|---|" + "---|" * (len(TASK_LABELS) + 1))
    avg_groups = {}  # config key -> [(name, average)]
    for name in sorted(zeroshot, key=sort_key):
        z = zeroshot[name]
        base, method, scheme, _, _ = parse_name(name)
        cells = " | ".join(f"{z['tasks'][t]['value']:.1f}" if t in z["tasks"] else "–" for t in TASK_LABELS)
        print(f"| {base} | {label(name)} | {scheme} | {cells} | {z['average']:.2f} |")
        if method != "BF16":
            avg_groups.setdefault(config_key(name), []).append((name, z["average"]))

    multi_seed = [runs for _, runs in sorted(avg_groups.items()) if len(runs) > 1]
    if multi_seed:
        print("\nZero-shot average over calibration seeds (mean ± sample std)")
        print("| Model | Method | Scheme | Seeds | Avg accuracy (%) | Δ vs BF16 |")
        print("|---|---|---|---|---|---|")
        for runs in multi_seed:
            first = runs[0][0]
            base, _, scheme, _, _ = parse_name(first)
            values = [v for _, v in runs]
            ref = zeroshot.get(base, {}).get("average")
            mean = statistics.mean(values)
            delta = f"{mean - ref:+.2f}" if ref is not None else "–"
            print(
                f"| {base} | {label(first, with_seed=False)} | {scheme} | {len(values)} "
                f"| {mean:.2f} ± {statistics.stdev(values):.2f} | {delta} |"
            )
