#!/usr/bin/env bash
# Speed and memory benchmark for the group B (W8A8 INT8) checkpoints, same workloads as group A.
#   RTN and SmoothQuant W8A8 (seed 0): same format, so they should run at the same speed (a check, as RTN/GPTQ/AWQ in group A).
#   RTN W8A8_STATIC: broken accuracy, but it shows what the per-token scale computation costs at run time.
# The BF16 and W4A16 results from group A are reused (same vLLM version and settings; the GPU probe in each log
# shows whether the GPU was shared during the run).
# Steps whose result already exists are skipped. Run from ~/Q_experiments with the vllm venv active, inside tmux:
#   bash scripts/run_speed_groupB.sh
set -e -o pipefail
mkdir -p logs
for size in 1B 3B; do
  for name in Llama-3.2-$size-RTN-W8A8 Llama-3.2-$size-SQ-W8A8 Llama-3.2-$size-RTN-W8A8_STATIC; do
    if [ -e "results/speed/$name.json" ]; then echo "skip $name (done)"; continue; fi
    echo "=== $name"
    python scripts/bench_vllm.py --model "models/$name" 2>&1 | tee "logs/bench_$name.log"
  done
done
