"""Collect every result JSON into one Markdown table, ready to paste into the report.

Usage (from ~/Q_experiments):  python scripts/summarize.py
"""
import json
from pathlib import Path

METHOD_ORDER = {"BF16": 0, "RTN": 1, "GPTQ": 2, "AWQ": 3}


def load(folder):
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(folder).glob("*.json"))}


ppl = load("results/ppl")
quant = load("results/quant")

rows = []
for run, r in ppl.items():
    q = quant.get(run)  # None for the BF16 baselines
    base_name = Path(q["model"]).name if q else run
    base_ppl = ppl.get(base_name, {}).get("ppl")
    rtn_ppl = ppl.get(f"{base_name}-RTN-{q['scheme']}", {}).get("ppl") if q else None
    method = q["method"].upper() if q else "BF16"

    delta = f"{r['ppl'] - base_ppl:+.3f}" if q and base_ppl else "–"
    # Share of RTN's perplexity loss that the method wins back (0% = RTN, 100% = BF16).
    recovered = (
        f"{(rtn_ppl - r['ppl']) / (rtn_ppl - base_ppl):.0%}"
        if q and rtn_ppl and base_ppl and method not in ("RTN",)
        else "–"
    )
    rows.append(
        (
            base_name,
            METHOD_ORDER.get(method, 9),
            f"| {base_name} | {method} | {q['scheme'] if q else 'BF16'} | {r['ppl']:.3f} | {delta} | {recovered} "
            f"| {q['size_on_disk_gib'] if q else '–'} | {q['quant_seconds'] if q else '–'} "
            f"| {q['peak_gpu_gib'] if q else '–'} |",
        )
    )

print("| Model | Method | Scheme | WikiText-2 PPL | Δ vs BF16 | RTN loss recovered | Size (GiB) | Quant time (s) | Quant peak GPU (GiB) |")
print("|---|---|---|---|---|---|---|---|---|")
for _, _, line in sorted(rows):
    print(line)
