#!/usr/bin/env bash
# Zero-shot accuracy for every group-A model: BF16, RTN, and GPTQ / AWQ with calibration seeds 0, 1 and 2.
# Run from ~/Q_experiments with the quant venv active:  bash scripts/run_zeroshot.sh
set -e -o pipefail
mkdir -p logs
for size in 1B 3B; do
  models="meta-llama/Llama-3.2-$size models/Llama-3.2-$size-RTN-W4A16_ASYM"
  for m in GPTQ AWQ; do
    models="$models models/Llama-3.2-$size-$m-W4A16_ASYM"
    models="$models models/Llama-3.2-$size-$m-W4A16_ASYM-seed1 models/Llama-3.2-$size-$m-W4A16_ASYM-seed2"
  done
  for model in $models; do
    name=$(basename "$model")
    # Skip models already evaluated, so the script can be restarted after an interruption.
    if [ -f "results/zeroshot/$name.json" ]; then echo "skip $name (already done)"; continue; fi
    python scripts/zeroshot_eval.py --model "$model" 2>&1 | tee "logs/zeroshot_$name.log"
  done
done
python scripts/summarize.py
