#!/usr/bin/env bash
# Group C, task 1: W4A4KV4 with GPTQ weights in SpinQuant's code, calibrated on our 128 C4 windows
# (the same ones as groups A and B; SpinQuant's default would be WikiText-2 train):
#   GPTQ                   no rotation: does GPTQ alone save 4-bit activations? (the papers say no)
#   QuaRot-GPTQ seed 0-2   QuaRot: random Hadamard rotations, then GPTQ. The rotation comes from --seed; the
#                          calibration windows stay the same, so the seeds differ only in the rotation.
# Same settings as run_groupC_rtn.sh, except GPTQ instead of RTN (SpinQuant's GPTQ defaults: 128 samples,
# dampening 0.01, no act-order, clipping search). Each run saves perplexity, quantization time and peak GPU memory.
# Before the first run (once):  quant venv: python scripts/export_calib.py ;  spinquant venv: bash scripts/patch_spinquant.sh
# Run from ~/Q_experiments, spinquant venv active, in tmux (about 1-1.5 hours):  bash scripts/run_groupC_gptq.sh
set -e -o pipefail
grep -q 'Q_CALIB_FILE' third_party/SpinQuant/utils/data_utils.py || { echo "run scripts/patch_spinquant.sh first"; exit 1; }
mkdir -p logs results/groupC outputs/spinquant

COMMON="--do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048
        --fp16 False --bf16 True --save_safetensors False --output_dir $PWD/outputs/spinquant"

run() {  # run <name> <calibration file> <SpinQuant arguments...>
  local name=$1 calib=$2
  shift 2
  if [ -e "results/groupC/$name.json" ]; then echo "skip $name (done)"; return; fi
  echo "=== $name"
  # shellcheck disable=SC2086
  python scripts/spinquant_eval.py --name "$name" --calib "$calib" -- $COMMON "$@" 2>&1 | tee "logs/groupC_$name.log" \
    | grep -E "wiki2 ppl|PPL .* quantization|Error"
}

for size in 1B 3B; do
  model=meta-llama/Llama-3.2-$size
  m=Llama-3.2-$size
  calib=data/calib/c4_${m}_128x2048_seed0.pt
  [ -f "$calib" ] || { echo "missing $calib: run scripts/export_calib.py in the quant venv"; exit 1; }
  if [ $size = 1B ]; then hd=64; else hd=128; fi
  W4A4KV4="--w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 --w_clip --a_asym --k_asym --v_asym
           --k_groupsize $hd --v_groupsize $hd"
  # shellcheck disable=SC2086
  run "${m}_spinquant_GPTQ_W4A4KV4" "$calib" --input_model $model $W4A4KV4 --seed 0
  for seed in 0 1 2; do
    # shellcheck disable=SC2086
    run "${m}_spinquant_QuaRot-GPTQ_W4A4KV4_seed$seed" "$calib" --input_model $model $W4A4KV4 \
      --rotate --rotate_mode hadamard --seed $seed
  done
done
