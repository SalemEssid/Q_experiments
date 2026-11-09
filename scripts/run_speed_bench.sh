#!/usr/bin/env bash
# Speed and memory benchmark for the BF16 baselines and every W4A16 checkpoint (seed 0 only:
# speed depends on the format, not on which calibration samples were used).
# Run from ~/Q_experiments with the vllm venv active:  bash scripts/run_speed_bench.sh
set -e -o pipefail
mkdir -p logs
for size in 1B 3B; do
  for model in meta-llama/Llama-3.2-$size \
               models/Llama-3.2-$size-RTN-W4A16_ASYM \
               models/Llama-3.2-$size-GPTQ-W4A16_ASYM \
               models/Llama-3.2-$size-AWQ-W4A16_ASYM; do
    name=$(basename "$model")
    python scripts/bench_vllm.py --model "$model" 2>&1 | tee "logs/bench_$name.log"
  done
done
