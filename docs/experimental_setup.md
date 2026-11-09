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
| lm-eval | 0.4.13 (zero-shot tasks) |

### vLLM environment (speed and memory measurements)

| Component | Version / setting |
|---|---|
| vLLM | 0.32.0+cu129 (the last release with a CUDA 12.x wheel; 0.33.0 is CUDA 13 only and cannot run on driver 535) |
| PyTorch | 2.13.0+cu129 (installed by vLLM) |
| transformers / compressed-tensors | 5.19.0 / 0.17.0 |
| Workaround 1 | `torchcodec` uninstalled. The PyPI build is for CUDA 13 and crashes `import vllm`. It is only used for audio/video input |
| Workaround 2 | `VLLM_USE_FLASHINFER_SAMPLER=0`. FlashInfer would compile its sampling kernels with the system nvcc 12.0, which is too old, so vLLM's PyTorch sampler is used instead. Same setting for every model, so the comparison stays fair |

## Quantization settings shared by all methods

| Setting | Value |
|---|---|
| Calibration data | 128 windows × 2048 tokens, random positions in long documents of C4 (train shard 00000), seed 0. Identical samples for every method (`scripts/calib_data.py`) |
| Layers not quantized | `lm_head` |
| W4A16 format | `W4A16_ASYM`: 4-bit integer weights with zero-point, group size 128, BF16 activations |
| Exact algorithm settings | saved by llm-compressor as `recipe.yaml` inside each checkpoint folder |

## Zero-shot accuracy protocol (`scripts/zeroshot_eval.py`)

- **Tool:** lm-evaluation-harness (lm-eval), Hugging Face backend, in the `quant` environment. The model is loaded exactly as for perplexity (BF16, `device_map="cuda"`) and passed to lm-eval as an object. Group C codebases will pass their own model objects to the same function.
- **Tasks:** ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande, BoolQ, the task set of the QuaRot, SpinQuant and DuQuant papers. 0-shot, lm-eval's default prompts.
- **How it is scored:** for each question the model scores every answer option by its log-probability, and the most likely option is its answer. Nothing is generated.
- **Metric:** `acc_norm` (accuracy with length-normalized probabilities) when the task has it (ARC-e, ARC-c, HellaSwag, PIQA), otherwise `acc` (WinoGrande, BoolQ). This is the rule used in QuaRot's evaluation code. The average is the plain mean over the six tasks.
- **Batch size 16** for every model. Batch 32 ran out of memory on the 3B: lm-eval computes log-probabilities over the full 128k vocabulary for every token in a batch.
- **Uncertainty:** lm-eval reports a standard error per task (about 0.5–1.5 points here, depending on the task size). Differences between methods smaller than that need care.

## Speed and memory protocol (`scripts/bench_vllm.py`)

All models run in vLLM with the same settings: `gpu_memory_utilization=0.85`, `max_model_len=2048`, prefix caching off, CUDA graphs and torch.compile at their defaults, greedy decoding with `ignore_eos` (every request produces exactly the requested number of tokens). Prompts are random token IDs of a fixed length, identical for every model.

| Workload | What it measures | Settings | Repeats |
|---|---|---|---|
| Latency, 1 request | **TTFT** (time to first token) and **TPOT** (time per output token). Memory-bound: favours small weights | 128 prompt tokens → 256 output tokens. TPOT = (time for 256 tokens − TTFT) / 255 | 3 warm-up, median of 10 |
| Prefill | Prompt processing speed. Compute-bound: favours fast math (INT8/INT4) | 32 requests × 1024 tokens → 1 token | 1 warm-up, median of 5 |
| Throughput under load | Output tokens/s when many users are served at once | 256 requests × 512 tokens → 256 tokens, all submitted together | 1 warm-up, median of 3 |

**Memory:** vLLM reserves the same share of GPU memory for every model, so `nvidia-smi` usage says nothing about the model. We report instead (1) the memory taken by the weights (vLLM log line "Model loading took X GiB") and (2) the **KV-cache capacity** in tokens left in the same budget. Smaller weights leave more room for the KV cache, so more requests can run at once.

**Shared GPU check:** a BF16 matmul probe runs before and after each benchmark. About 110 TFLOPS means the physical GPU was idle. A lower value means another VM was using it, and the run should be repeated.

## Notes that affect the comparison

- **Shared GPU:** the physical A40 may be used by another VM at the same time, which slows our runs. Speed results are repeated several times and the median is reported. The BF16 matmul figure above is the reference for an idle GPU.
- **No FP8 on Ampere:** 8-bit weight-and-activation methods (W8A8, e.g. SmoothQuant) use INT8.
