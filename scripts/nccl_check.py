"""Does NCCL (the GPU communication library that SpinQuant's torchrun scripts start) work on this vGPU?

SpinQuant calls dist.init_process_group("nccl") and dist.barrier() even with one GPU. This does the same, plus one
all_reduce, in a few seconds. Usage (spinquant venv, from ~/Q_experiments):
  torchrun --nproc_per_node=1 scripts/nccl_check.py                          # default settings
  NCCL_CUMEM_ENABLE=0 torchrun --nproc_per_node=1 scripts/nccl_check.py      # without NCCL's cuMem memory
Result on the A40-24Q (2026-11-10): both fail. NCCL 2.26.2 calls cudaMemPoolCreate at start-up (init.cc:416), which
the vGPU does not support ("operation not supported"). Hence scripts/patch_spinquant.sh switches SpinQuant to gloo.
"""
import os

import torch
import torch.distributed as dist

torch.cuda.set_device(0)
dist.init_process_group(backend="nccl")
dist.barrier()
x = torch.ones(4, device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()
print(f"NCCL OK (all_reduce gave {x[0].item():.0f}), NCCL_CUMEM_ENABLE={os.environ.get('NCCL_CUMEM_ENABLE', 'not set')}")
dist.destroy_process_group()
