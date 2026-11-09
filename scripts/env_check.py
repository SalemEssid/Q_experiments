"""Step 1: check that PyTorch can use the GPU on this server.

Run:  python env_check.py
Paste the full output back; we will record it in the report's "Experimental setup" section.
"""
import platform
import time

import torch

print("Python        :", platform.python_version())
print("PyTorch       :", torch.__version__)
print("CUDA (torch)  :", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the GPU. Paste this output so we can fix the install.")

props = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info()
print("GPU           :", props.name)
print("Compute cap.  :", f"{props.major}.{props.minor}")
print(f"GPU memory    : {free / 2**30:.1f} GiB free / {total / 2**30:.1f} GiB total")
print("BF16 support  :", torch.cuda.is_bf16_supported())

# 1) Raw speed: a large BF16 matrix multiply, the core operation inside every LLM layer.
#    This gives a reference number to tell whether the shared GPU is busy later on.
n = 8192
a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
for _ in range(3):  # warm-up: the first calls are slower (kernel selection, clocks ramping up)
    a @ b
torch.cuda.synchronize()  # GPU work is asynchronous; wait for it before reading the clock
iters = 20
start = time.perf_counter()
for _ in range(iters):
    a @ b
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
tflops = 2 * n**3 * iters / elapsed / 1e12  # an n x n matmul costs 2*n^3 floating-point operations
print(f"BF16 matmul   : {tflops:.1f} TFLOPS (A40 theoretical peak is ~150)")

# 2) Runtime compiler check: vLLM and several quantization kernels generate GPU code on the fly
#    with Triton. This needs a working C compiler and Python headers on the server.
try:
    f = torch.compile(lambda x: torch.nn.functional.gelu(x) * 2)
    f(torch.randn(1024, device="cuda"))
    print("torch.compile : OK")
except Exception as e:
    print("torch.compile : FAILED ->", type(e).__name__, str(e)[:300])
