#!/usr/bin/env bash
# Group B main runs: W8A8 with per-token activation scales, both models.
#   RTN (no calibration, one run), SmoothQuant α=0.5 with calibration seeds 0/1/2, SmoothQuant α=0.8 (seed 0).
#   For each: WikiText-2 perplexity, weights-only perplexity, zero-shot tasks.
# Then a short static-scale check on the 3B (does it have massive activations too?).
# Steps whose result already exists are skipped, so the script can be restarted after an interruption.
# Run from ~/Q_experiments with the quant venv active, inside tmux:  bash scripts/run_groupB.sh
set -e -o pipefail
mkdir -p logs

step() {  # step <file or folder that exists when the step is done> <log name> <command...>
  local done=$1 log=$2
  shift 2
  if [ -e "$done" ]; then echo "skip $log (done)"; return; fi
  echo "=== $log"
  "$@" 2>&1 | tee "logs/$log.log"
}

runs() {  # the W8A8 run names for one model size
  local m=Llama-3.2-$1
  echo "$m-RTN-W8A8 $m-SQ-W8A8 $m-SQ-W8A8-seed1 $m-SQ-W8A8-seed2 $m-SQ-W8A8-alpha0.8"
}

# 1. Quantize, then perplexity (fast): results for both models come early.
for size in 1B 3B; do
  base=meta-llama/Llama-3.2-$size
  m=Llama-3.2-$size
  step "models/$m-RTN-W8A8" "quant_$m-RTN-W8A8" python scripts/quantize.py --model $base --method rtn --scheme W8A8
  step "models/$m-SQ-W8A8" "quant_$m-SQ-W8A8" python scripts/quantize.py --model $base --method sq --scheme W8A8
  for seed in 1 2; do
    step "models/$m-SQ-W8A8-seed$seed" "quant_$m-SQ-W8A8-seed$seed" \
      python scripts/quantize.py --model $base --method sq --scheme W8A8 --seed $seed
  done
  step "models/$m-SQ-W8A8-alpha0.8" "quant_$m-SQ-W8A8-alpha0.8" \
    python scripts/quantize.py --model $base --method sq --scheme W8A8 --alpha 0.8
  for name in $(runs $size); do
    step "results/ppl/$name.json" "ppl_$name" python scripts/ppl_eval.py --model models/$name
    step "results/ppl_weights_only/$name.json" "ppl_wonly_$name" python scripts/ppl_eval.py --model models/$name --weights-only
  done
done

# 2. Static-scale check on the 3B: RTN with one fixed activation scale per layer, the same with BF16 inputs in
#    down_proj, and the activation measurements of the BF16 model.
m=Llama-3.2-3B
step "models/$m-RTN-W8A8_STATIC" "quant_$m-RTN-W8A8_STATIC" \
  python scripts/quantize.py --model meta-llama/$m --method rtn --scheme W8A8_STATIC
step "results/ppl/$m-RTN-W8A8_STATIC.json" "ppl_$m-RTN-W8A8_STATIC" python scripts/ppl_eval.py --model models/$m-RTN-W8A8_STATIC
step "results/ppl_act_diag/$m-RTN-W8A8_STATIC-actoff-down_proj.json" "ppl_diag_$m-RTN-W8A8_STATIC" \
  python scripts/ppl_eval.py --model models/$m-RTN-W8A8_STATIC --act-off-in down_proj
step "results/act_stats/$m.json" "act_stats_$m" python scripts/act_stats.py --model meta-llama/$m

# 3. Zero-shot tasks (slow: about 2 hours in total).
for size in 1B 3B; do
  for name in $(runs $size); do
    step "results/zeroshot/$name.json" "zeroshot_$name" python scripts/zeroshot_eval.py --model models/$name
  done
done

python scripts/summarize.py
