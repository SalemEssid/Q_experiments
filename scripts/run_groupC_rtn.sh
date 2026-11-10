#!/usr/bin/env bash
# Group C, step 3: W4A4KV4 with plain rounding (RTN, no calibration data needed), in SpinQuant's code:
#   BF16                  the same codebase's own baseline (group C differences are measured against it)
#   RTN                   no rotation: the reference, expected to break
#   QuaRot-RTN seed 0-2   QuaRot's random Hadamard rotations (R1, R2 folded into the weights; online Hadamards
#                         before down_proj and on the keys), then the same RTN. The random rotation is drawn from
#                         --seed (SpinQuant's --rotation_seed is never used), so 3 seeds = 3 different rotations.
# Settings, the same for every group C method (decision log 2026-11-10), as in SpinQuant's own scripts:
#   weights      4 bit, symmetric, one scale per output channel, clipping ratio searched (--w_clip)
#   activations  4 bit, asymmetric, one scale per token (o_proj: per head, fixed in the code); lm_head stays BF16
#   KV cache     4 bit, asymmetric, one scale per token and head (group size = head_dim: 64 for 1B, 128 for 3B;
#                SpinQuant accepts nothing else for the keys)
# Steps whose result exists are skipped. Run from ~/Q_experiments, spinquant venv active, in tmux (about 40 min):
#   bash scripts/run_groupC_rtn.sh
set -e -o pipefail
grep -q 'init_process_group(backend="gloo"' third_party/SpinQuant/ptq.py || { echo "run scripts/patch_spinquant.sh first"; exit 1; }
mkdir -p logs results/groupC outputs/spinquant
cd third_party/SpinQuant

COMMON="--do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048
        --fp16 False --bf16 True --save_safetensors False --output_dir ../../outputs/spinquant"

run() {  # run <name> <model> <SpinQuant arguments...>
  local name=$1 model=$2
  shift 2
  local out=../../results/groupC/$name.json log=../../logs/groupC_$name.log
  if [ -e "$out" ]; then echo "skip $name (done)"; return; fi
  echo "=== $name"
  local start=$(date +%s)
  # shellcheck disable=SC2086
  torchrun --nnodes=1 --nproc_per_node=1 ptq.py --input_model "$model" $COMMON "$@" 2>&1 | tee "$log" \
    | grep -E "wiki2 ppl|Error" || true
  python ../../scripts/groupC_result.py --code-dir . --log "$log" --out "$out" --name "$name" \
    --seconds $(( $(date +%s) - start )) --args "$*"
}

for size in 1B 3B; do
  model=meta-llama/Llama-3.2-$size
  m=Llama-3.2-$size
  if [ $size = 1B ]; then hd=64; else hd=128; fi
  W4A4KV4="--w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 --w_rtn --w_clip --a_asym --k_asym --v_asym
           --k_groupsize $hd --v_groupsize $hd"
  run "${m}_spinquant_BF16" $model --w_bits 16 --a_bits 16 --k_bits 16 --v_bits 16
  # shellcheck disable=SC2086
  run "${m}_spinquant_RTN_W4A4KV4" $model $W4A4KV4 --seed 0
  for seed in 0 1 2; do
    # shellcheck disable=SC2086
    run "${m}_spinquant_QuaRot-RTN_W4A4KV4_seed$seed" $model $W4A4KV4 --rotate --rotate_mode hadamard --seed $seed
  done
done
