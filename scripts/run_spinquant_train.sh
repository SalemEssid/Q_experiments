#!/usr/bin/env bash
# Group C, task 2: SpinQuant = learned rotations, then GPTQ.
#   1. optimize_rotation.py learns R1 and R2 (Cayley SGD, 100 steps, lr 1.5, cosine) on a model whose activations
#      and KV cache are 4-bit and whose weights stay 16-bit (W16A4KV4): the authors' recipe when GPTQ follows.
#      Data: our 800 C4 windows (the authors use 800 WikiText-2 windows). The authors' 8 GPUs x batch 1 become
#      1 GPU x batch 1 x 8 gradient-accumulation steps: the same 8 windows per step and 100 steps.
#      One process without torchrun: NCCL cannot start on this vGPU, and with one process the Trainer does not
#      need it (patch_spinquant.sh switches torch.distributed to gloo).
#   2. spinquant_eval.py: the learned rotation + GPTQ W4A4KV4 on our 128 C4 windows, same settings as QuaRot-GPTQ.
# The training time and peak GPU memory are added to the result, since they are part of SpinQuant's cost.
# Choose sizes and seeds with SIZES and SEEDS (seed = the random initial rotation), e.g. SIZES=1B SEEDS="0 1 2".
# Run from ~/Q_experiments, spinquant venv active, in tmux:  bash scripts/run_spinquant_train.sh
set -e -o pipefail
grep -q 'Q_ROTATION_SEED' third_party/SpinQuant/optimize_rotation.py || { echo "run scripts/patch_spinquant.sh first"; exit 1; }
ROOT=$PWD
mkdir -p logs results/groupC outputs/spinquant/rotation
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29512 RANK=0 WORLD_SIZE=1

for size in ${SIZES:-1B 3B}; do
  model=meta-llama/Llama-3.2-$size
  m=Llama-3.2-$size
  if [ $size = 1B ]; then hd=64; else hd=128; fi
  for seed in ${SEEDS:-0}; do
    name=${m}_spinquant_SpinQuant-GPTQ_W4A4KV4_seed$seed
    rot=$ROOT/outputs/spinquant/rotation/${m}_W16A4KV4_seed$seed
    if [ -e "results/groupC/$name.json" ]; then echo "skip $name (done)"; continue; fi

    if [ ! -f "$rot/R.bin" ]; then
      echo "=== train rotation: $m seed $seed"
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "logs/gpumem_train_${m}_seed$seed.txt" &
      smi=$!
      start=$(date +%s)
      (cd third_party/SpinQuant && Q_CALIB_FILE=$ROOT/data/calib/c4_${m}_800x2048_seed0.pt Q_ROTATION_SEED=$seed \
        python optimize_rotation.py \
          --input_model $model --output_rotation_path "$rot" \
          --output_dir "$ROOT/outputs/spinquant/train_${m}_seed$seed" --logging_dir "$ROOT/logs/tb_${m}_seed$seed" \
          --model_max_length 2048 --fp16 False --bf16 True --log_on_each_node False \
          --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --logging_steps 1 \
          --learning_rate 1.5 --weight_decay 0. --lr_scheduler_type cosine --gradient_checkpointing True \
          --save_safetensors False --save_strategy no --report_to none --max_steps 100 \
          --w_bits 16 --a_bits 4 --k_bits 4 --v_bits 4 --w_clip --a_asym --k_asym --v_asym \
          --k_groupsize $hd --v_groupsize $hd) 2>&1 | tee "logs/train_${m}_seed$seed.log" \
        | grep -E "'loss'|Error|error" || true
      seconds=$(( $(date +%s) - start ))
      kill $smi 2>/dev/null || true
      [ -f "$rot/R.bin" ] || { echo "training failed: see logs/train_${m}_seed$seed.log"; exit 1; }
      peak=$(sort -n "logs/gpumem_train_${m}_seed$seed.txt" | tail -1)
      printf '{"rotation_training_seconds": %s, "rotation_training_peak_gpu_mib": %s, "rotation_file": "%s"}\n' \
        "$seconds" "${peak:-null}" "$rot/R.bin" > "$rot/training.json"
      echo "rotation trained in $seconds s, peak GPU memory ${peak} MiB (nvidia-smi)"
    fi

    echo "=== $name"
    python scripts/spinquant_eval.py --name "$name" --calib data/calib/c4_${m}_128x2048_seed0.pt --extra "$rot/training.json" -- \
      --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 \
      --fp16 False --bf16 True --save_safetensors False --output_dir "$ROOT/outputs/spinquant" \
      --input_model $model --w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 --w_clip --a_asym --k_asym --v_asym \
      --k_groupsize $hd --v_groupsize $hd --rotate --optimized_rotation_path "$rot/R.bin" --seed $seed \
      2>&1 | tee "logs/groupC_$name.log" | grep -E "wiki2 ppl|PPL .* quantization|Error"
  done
done
