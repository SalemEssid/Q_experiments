#!/usr/bin/env bash
# Group C, task 3: zero-shot accuracy for the group C models (the six lm-eval tasks of groups A and B, batch 16,
# scored by the same scripts/zeroshot_eval.py).
# spinquant_eval.py --zeroshot rebuilds each model exactly as in its perplexity run: same flags, same rotation seed,
# same C4 calibration windows, SpinQuant's saved learned rotation (R.bin, no retraining). It measures perplexity again,
# as a reproducibility check against results/groupC, then the zero-shot tasks. Results go to results/groupC_zeroshot/,
# so the perplexity results stay untouched.
# Order = priority, so the most important numbers arrive first:
#   1. BF16 (does this code reproduce our BF16 zero-shot?), QuaRot-GPTQ seed 0, SpinQuant seed 0    (1B, then 3B)
#   2. QuaRot-GPTQ seeds 1 and 2 (seed spread, as for GPTQ and AWQ in group A)
#   3. QuaRot-RTN seeds 0-2, and RTN without rotation (a broken model: chance-level reference)
# GPTQ without rotation (also broken, and slow to quantize) is left out.
# Runs whose result exists are skipped, so the script can be restarted. A failed run stops the script.
# Run from ~/Q_experiments, spinquant venv active, in tmux:  bash scripts/run_groupC_zeroshot.sh
set -e -o pipefail
python -c "import lm_eval" 2>/dev/null || { echo "run scripts/setup_spinquant_lmeval.sh first"; exit 1; }
grep -q 'Q_CALIB_FILE' third_party/SpinQuant/utils/data_utils.py || { echo "run scripts/patch_spinquant.sh first"; exit 1; }
ROOT=$PWD
OUT=$ROOT/results/groupC_zeroshot
mkdir -p logs "$OUT"

COMMON="--do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048
        --fp16 False --bf16 True --save_safetensors False --output_dir $ROOT/outputs/spinquant"

run() {  # run <name> <calibration file, or - for none> <SpinQuant arguments...>
  local name=$1 calib=$2
  shift 2
  if [ -e "$OUT/$name.json" ]; then echo "skip $name (done)"; return; fi
  echo "=== $name (start $(date +%H:%M))"
  local calib_arg=()
  [ "$calib" = - ] || calib_arg=(--calib "$calib")
  # shellcheck disable=SC2086
  python scripts/spinquant_eval.py --name "$name" "${calib_arg[@]}" --zeroshot --batch-size 16 --out-dir "$OUT" \
    -- $COMMON "$@" 2>&1 | tee "logs/groupC_zeroshot_$name.log" | grep -E "wiki2 ppl|zero-shot average|Error"
}

setup() {  # setup <1B|3B>: sets model, m, calib, rot, W4A4KV4 (GPTQ weights) and W4A4KV4_RTN
  model=meta-llama/Llama-3.2-$1
  m=Llama-3.2-$1
  if [ "$1" = 1B ]; then hd=64; else hd=128; fi
  calib=$ROOT/data/calib/c4_${m}_128x2048_seed0.pt
  rot=$ROOT/outputs/spinquant/rotation/${m}_W16A4KV4_seed0/R.bin
  [ -f "$calib" ] || { echo "missing $calib"; exit 1; }
  [ -f "$rot" ] || { echo "missing $rot (SpinQuant's learned rotation)"; exit 1; }
  # The same settings as run_groupC_rtn.sh / run_groupC_gptq.sh / run_spinquant_train.sh
  W4A4KV4="--w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 --w_clip --a_asym --k_asym --v_asym
           --k_groupsize $hd --v_groupsize $hd"
  W4A4KV4_RTN="$W4A4KV4 --w_rtn"
}

# shellcheck disable=SC2086
{
  # 1. the main numbers
  for size in 1B 3B; do
    setup $size
    run "${m}_spinquant_BF16" - --input_model $model --w_bits 16 --a_bits 16 --k_bits 16 --v_bits 16
    run "${m}_spinquant_QuaRot-GPTQ_W4A4KV4_seed0" "$calib" --input_model $model $W4A4KV4 \
      --rotate --rotate_mode hadamard --seed 0
    run "${m}_spinquant_SpinQuant-GPTQ_W4A4KV4_seed0" "$calib" --input_model $model $W4A4KV4 \
      --rotate --optimized_rotation_path "$rot" --seed 0
  done
  # 2. seed spread of QuaRot-GPTQ
  for size in 1B 3B; do
    setup $size
    for seed in 1 2; do
      run "${m}_spinquant_QuaRot-GPTQ_W4A4KV4_seed$seed" "$calib" --input_model $model $W4A4KV4 \
        --rotate --rotate_mode hadamard --seed $seed
    done
  done
  # 3. RTN weights
  for size in 1B 3B; do
    setup $size
    for seed in 0 1 2; do
      run "${m}_spinquant_QuaRot-RTN_W4A4KV4_seed$seed" - --input_model $model $W4A4KV4_RTN \
        --rotate --rotate_mode hadamard --seed $seed
    done
    run "${m}_spinquant_RTN_W4A4KV4" - --input_model $model $W4A4KV4_RTN --seed 0
  done
}
echo "all group C zero-shot runs done"
