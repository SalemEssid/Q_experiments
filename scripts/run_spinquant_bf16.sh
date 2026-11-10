#!/usr/bin/env bash
# Group C, step 2: BF16 perplexity with SpinQuant's own code (no rotation, nothing quantized).
# Fairness rule 2: it must reproduce our BF16 perplexity (9.756 / 7.817) before we trust its quantized results.
# SpinQuant's evaluation: WikiText-2 test joined with "\n\n", no BOS token, non-overlapping 2048-token windows,
# the same protocol as scripts/ppl_eval.py.
# Run from ~/Q_experiments with the spinquant venv active:  bash scripts/run_spinquant_bf16.sh
set -e -o pipefail
# NCCL cannot start on this vGPU, so SpinQuant must use gloo (scripts/patch_spinquant.sh; this checks it was applied).
grep -q 'init_process_group(backend="gloo"' third_party/SpinQuant/ptq.py || { echo "run scripts/patch_spinquant.sh first"; exit 1; }
mkdir -p logs outputs/spinquant
cd third_party/SpinQuant
for size in 1B 3B; do
  echo "=== Llama-3.2-$size BF16 (SpinQuant code)"
  torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
    --input_model meta-llama/Llama-3.2-$size \
    --output_dir ../../outputs/spinquant \
    --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 \
    --fp16 False --bf16 True --save_safetensors False \
    --w_bits 16 --a_bits 16 --k_bits 16 --v_bits 16 \
    2>&1 | tee ../../logs/spinquant_bf16_Llama-3.2-$size.log | grep -E "wiki2 ppl|Error|error" || true
done
