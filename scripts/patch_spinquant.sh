#!/usr/bin/env bash
# Our changes to the official SpinQuant code in third_party/SpinQuant (listed in docs/experimental_setup.md).
# Safe to run more than once. `git -C third_party/SpinQuant diff` shows every change.
# Run from ~/Q_experiments:  bash scripts/patch_spinquant.sh
set -e -o pipefail
cd third_party/SpinQuant

# Change 1: gloo instead of NCCL for torch.distributed.
#   NCCL 2.26 (shipped with PyTorch 2.7) cannot start on the A40-24Q vGPU: cudaMemPoolCreate fails with
#   "operation not supported" (NCCL init.cc:416), and no NCCL setting skips it. SpinQuant starts a process group even
#   on one GPU, but with one process nothing is exchanged: the code only asks for the rank and waits at barriers.
#   gloo (PyTorch's CPU backend) does the same and cannot change any result.
for f in ptq.py optimize_rotation.py; do
  sed -i 's/dist.init_process_group(backend="nccl"/dist.init_process_group(backend="gloo"/' "$f"
  grep -q 'init_process_group(backend="gloo"' "$f" || { echo "patch failed in $f"; exit 1; }
done

echo "SpinQuant changes vs commit $(git rev-parse --short HEAD):"
git diff --stat
