#!/usr/bin/env bash
# Calibration-seed study: is the GPTQ vs AWQ gap bigger than the effect of picking
# different calibration samples? Seed 0 (our standard set) is already done; this adds seeds 1 and 2.
# Run from ~/Q_experiments with the quant venv active:  bash scripts/run_seed_study.sh
set -e
for model in Llama-3.2-1B Llama-3.2-3B; do
  for seed in 1 2; do
    for m in gptq awq; do
      python scripts/quantize.py --method $m --model meta-llama/$model --seed $seed
      python scripts/ppl_eval.py --model models/$model-${m^^}-W4A16_ASYM-seed$seed
    done
  done
done
python scripts/summarize.py
