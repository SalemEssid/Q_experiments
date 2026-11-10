#!/usr/bin/env bash
# Where does the W8A8 activation error come from? For each group of layers, leave only that group's input
# activations in BF16 (everything else stays W8A8) and re-measure WikiText-2 perplexity. The bigger the drop,
# the more of the activation error that group causes.
#   q_proj,k_proj,v_proj and gate_proj,up_proj: inputs that SmoothQuant smooths (they follow an RMSNorm)
#   o_proj and down_proj: inputs that SmoothQuant cannot smooth on Llama (no norm right before them)
# Run from ~/Q_experiments with the quant venv active:
#   bash scripts/run_act_diag.sh models/Llama-3.2-1B-RTN-W8A8 models/Llama-3.2-1B-SQ-W8A8
set -e -o pipefail
for model in "$@"; do
  echo "== $model"
  for group in q_proj,k_proj,v_proj o_proj gate_proj,up_proj down_proj; do
    python scripts/ppl_eval.py --model "$model" --act-off-in "$group" 2>&1 | grep -E "OFF in|\"ppl\""
  done
done
