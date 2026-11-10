#!/usr/bin/env bash
# Group C, step 1: the SpinQuant environment. Also used for "QuaRot via SpinQuant" and the RTN W4A4 reference.
#   ~/venvs/spinquant   PyTorch 2.6.0 (CUDA 12.4 build; runs on driver 535 like the CUDA 12.6 build in ~/venvs/quant)
#                       + SpinQuant's pinned versions (transformers 4.44.2, accelerate 0.34.2, datasets 2.20.0)
#                       + fast-hadamard-transform from its prebuilt wheel: nothing is compiled, so the
#                         nvcc 12.0 / gcc 13.2 problem does not apply.
#   third_party/SpinQuant   the official code at a fixed commit (recorded in the log).
# Run from ~/Q_experiments with no venv active (about 10 minutes, mostly downloads):
#   bash scripts/setup_spinquant_env.sh 2>&1 | tee logs/setup_spinquant_env.log
set -e -o pipefail
SPINQUANT_COMMIT=8f47aa3f00e8662caf1a484153920a07e5281c3a  # main branch, 2025-02-14
VENV=$HOME/venvs/spinquant
FHT_RELEASE=https://github.com/Dao-AILab/fast-hadamard-transform/releases/download/v1.1.0.post2
mkdir -p logs third_party/wheels

# The prebuilt fast-hadamard-transform wheels used below are for Python 3.11.
python3 -c 'import sys; assert sys.version_info[:2] == (3, 11), f"need Python 3.11, found {sys.version}"'

[ -d "$VENV" ] || python3 -m venv "$VENV"
# datasets 2.20 keeps its own cache, so it never reads files written by the newer datasets in ~/venvs/quant.
grep -q HF_DATASETS_CACHE "$VENV/bin/activate" || \
  echo 'export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets_spinquant' >> "$VENV/bin/activate"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

[ -d third_party/SpinQuant ] || git clone https://github.com/facebookresearch/SpinQuant.git third_party/SpinQuant
git -C third_party/SpinQuant checkout "$SPINQUANT_COMMIT"
pip install -r third_party/SpinQuant/requirement.txt "numpy<2"

# Pick the wheel that matches this PyTorch (version, CUDA major version, C++ ABI), as the package's own setup.py does.
WHEEL=$(python -c '
import torch
abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
cuda = torch.version.cuda.split(".")[0]
ver = ".".join(torch.__version__.split(".")[:2])
print(f"fast_hadamard_transform-1.1.0+cu{cuda}torch{ver}cxx11abi{abi}-cp311-cp311-linux_x86_64.whl")')
echo "wheel: $WHEEL"
[ -f "third_party/wheels/$WHEEL" ] || curl -fL -o "third_party/wheels/$WHEEL" "$FHT_RELEASE/${WHEEL//+/%2B}"
pip install "third_party/wheels/$WHEEL"

echo "=== check"
echo "SpinQuant commit: $(git -C third_party/SpinQuant rev-parse HEAD)"
python - <<'EOF'
import accelerate, datasets, torch, transformers
import fast_hadamard_transform as fht

print("torch", torch.__version__, "| CUDA (torch)", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "| accelerate", accelerate.__version__, "| datasets", datasets.__version__)

# The online Hadamard rotation of down_proj's input (size 8192) must equal a multiply by the explicit matrix.
n = 8192
x = torch.randn(16, n, device="cuda", dtype=torch.float16)
y = fht.hadamard_transform(x, scale=n ** -0.5)
H = torch.ones(1, 1, device="cuda")
while H.shape[0] < n:  # Sylvester construction
    H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
ref = (x.float() @ H) * n ** -0.5
err = ((y.float() - ref).norm() / ref.norm()).item()
print(f"fast-hadamard-transform, n={n}: relative error vs explicit Hadamard {err:.1e} (FP16 rounding: about 1e-4 to 1e-3)")
assert err < 1e-2, "fast-hadamard-transform gives wrong results"
print("OK")
EOF
