# Experimental setup

All experiments run on the same machine and software stack, so results are directly comparable.

## Hardware

| Component | Value |
|---|---|
| GPU | NVIDIA A40-24Q: a virtual GPU (24 GB slice of a 48 GB A40, shared with another VM) |
| Architecture | Ampere, compute capability 8.6: supports BF16, INT8 and INT4 tensor cores, **no FP8** |
| GPU memory | 23.8 GiB visible, 23.2 GiB free after the CUDA context is created |
| Measured BF16 matmul | 109.9 TFLOPS (8192×8192, about 73% of the 150 TFLOPS theoretical peak), 2026-11-09 |
| CPU / RAM | 32 cores / 62 GiB, no swap |
| OS | Ubuntu 23.10 |

## Software

| Component | Version |
|---|---|
| NVIDIA driver | 535.230.02 (supports CUDA up to 12.2; tied to the vGPU host, cannot be upgraded) |
| CUDA toolkit (nvcc) | 12.0 |
| gcc | 13.2 |
| Python | 3.11.6 (venv) |
| PyTorch | 2.14.1+cu126 |
| transformers | 5.21.0 (set by llm-compressor; baselines re-run with this version) |
| llm-compressor | 0.15.0 (GPTQ, AWQ, SmoothQuant) |
| compressed-tensors | 0.20.1 (checkpoint format) |

## Quantization settings shared by all methods

| Setting | Value |
|---|---|
| Calibration data | 128 windows × 2048 tokens, random positions in long documents of C4 (train shard 00000), seed 0. Identical samples for every method (`scripts/calib_data.py`) |
| Layers not quantized | `lm_head` |
| W4A16 format | `W4A16_ASYM`: 4-bit integer weights with zero-point, group size 128, BF16 activations |
| Exact algorithm settings | saved by llm-compressor as `recipe.yaml` inside each checkpoint folder |

## Notes that affect the comparison

- **Shared GPU:** the physical A40 may be used by another VM at the same time, which slows our runs. Speed results are repeated several times and the median is reported. The BF16 matmul figure above is the reference for an idle GPU.
- **No FP8 on Ampere:** 8-bit weight-and-activation methods (W8A8, e.g. SmoothQuant) use INT8.
