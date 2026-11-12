#!/usr/bin/env bash
# Group C, task 3, step 1: add lm-eval (zero-shot tasks) to the spinquant venv without changing its pinned versions.
#   lm-eval 0.4.13 = the version of groups A and B. It needs transformers >= 4.30 (we have 4.44.2), datasets >= 2.16
#   (2.20.0) and accelerate >= 0.26 (0.34.2); its Hugging Face wrapper only switches to newer code from transformers
#   4.56 on. The six task datasets are parquet repos (no loading scripts), so datasets 2.20 can read them. They are
#   downloaded once into the spinquant datasets cache (HF_DATASETS_CACHE, set in the venv's activate script).
#   The versions already installed are repeated in the pip command, so pip cannot upgrade them as a side effect.
# Run from ~/Q_experiments with the spinquant venv active:  bash scripts/setup_spinquant_lmeval.sh
set -e -o pipefail
[ "$(basename "${VIRTUAL_ENV:-}")" = spinquant ] || { echo "activate ~/venvs/spinquant first"; exit 1; }

pip install "lm-eval==0.4.13" "peft==0.13.2" \
  "torch==2.7.1" "transformers==4.44.2" "accelerate==0.34.2" "datasets==2.20.0" "numpy<2" "pyarrow<26"
pip check || echo "WARNING: pip check reports a conflict (see above); send it to me before running the zero-shot script"

python - <<'EOF'
import accelerate, datasets, numpy, torch, transformers
import lm_eval
from lm_eval.models.huggingface import HFLM  # the wrapper used by scripts/zeroshot_eval.py
import fast_hadamard_transform  # still loads (SpinQuant's online Hadamards)
print("torch", torch.__version__, "| transformers", transformers.__version__, "| accelerate", accelerate.__version__,
      "| datasets", datasets.__version__, "| numpy", numpy.__version__, "| lm_eval", lm_eval.__version__)
assert torch.__version__.startswith("2.7.1"), "torch changed"
assert transformers.__version__ == "4.44.2" and datasets.__version__ == "2.20.0", "pinned versions changed"
print("OK")
EOF
