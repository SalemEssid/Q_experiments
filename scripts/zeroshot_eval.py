"""Zero-shot accuracy on six common-sense and reasoning tasks with lm-evaluation-harness (lm-eval).

Same task set as the QuaRot, SpinQuant and DuQuant papers. The model is loaded exactly as in ppl_eval.py
and handed to lm-eval as an object. The research codebases (group C) can later pass their own quantized
model objects to evaluate_zeroshot() in the same way, so every method is scored by the same code.

How a zero-shot multiple-choice task is scored: for each question, the model computes the probability
of every answer option; the option with the highest (length-normalized, for acc_norm) probability is
its answer. No examples are shown first (0-shot) and nothing is generated.

Usage (from ~/Q_experiments, quant venv):
  python scripts/zeroshot_eval.py --model meta-llama/Llama-3.2-1B
  python scripts/zeroshot_eval.py --model models/Llama-3.2-1B-GPTQ-W4A16_ASYM
"""
import argparse
import json
import time
from pathlib import Path

import lm_eval
import torch
import transformers
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "boolq"]


def _percent(value):
    try:
        return round(100 * float(value), 2)
    except (TypeError, ValueError):  # lm-eval writes "N/A" when a stderr is not available
        return None


def evaluate_zeroshot(model, tokenizer, batch_size=32, tasks=TASKS):
    """Score a loaded model. Returns ({task: {metric, value, stderr}} in %, average in %)."""
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    out = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, log_samples=False)
    scores = {}
    for task in tasks:
        r = out["results"][task]
        # Same rule as QuaRot's evaluation code: normalized accuracy if the task has it, else accuracy.
        metric = "acc_norm" if "acc_norm,none" in r else "acc"
        scores[task] = {
            "metric": metric,
            "value": _percent(r[f"{metric},none"]),
            "stderr": _percent(r.get(f"{metric}_stderr,none")),
        }
    average = round(sum(s["value"] for s in scores.values()) / len(scores), 2)
    return scores, average


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local folder")
    # 16, not 32: lm-eval computes log-probabilities over the whole 128k vocabulary for every token in a
    # batch (about 4.5 GiB at batch 16 for the longest questions), which ran out of memory on the 3B at 32.
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", default="results/zeroshot")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    # W8A8 models: number of layers whose input activations are rounded to INT8 (0 for BF16 and W4A16).
    act_layers = sum(
        1 for m in model.modules()
        if getattr(getattr(m, "quantization_scheme", None), "input_activations", None)
    )
    print("Layers with activation quantization:", act_layers)

    start = time.perf_counter()
    scores, average = evaluate_zeroshot(model, tokenizer, args.batch_size)
    elapsed = time.perf_counter() - start

    result = {
        "model": args.model,
        "num_fewshot": 0,
        "batch_size": args.batch_size,
        "activation_quant_layers": act_layers,
        "tasks": scores,
        "average": average,
        "eval_seconds": round(elapsed, 1),
        "lm_eval": lm_eval.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    out = Path(args.out_dir) / f"{Path(args.model).name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("Saved to", out)


if __name__ == "__main__":
    main()
