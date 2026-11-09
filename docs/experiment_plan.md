# Experiment plan

Models: Llama-3.2-1B (fast iteration) and Llama-3.2-3B (main results), base versions, BF16 baseline.

The methods target different formats, so they are compared **within a group**. Comparisons across groups need care (W4A16 saves memory traffic; W8A8 and W4A4 also save arithmetic).

## Group A: weight-only 4-bit (W4A16, group size 128, asymmetric)

Goal: smaller model, faster single-user generation.

| Method | Implementation | Status |
|---|---|---|
| RTN (reference) | llm-compressor | perplexity done (1B, 3B) |
| GPTQ | llm-compressor | perplexity done (1B, 3B, 3 seeds) |
| AWQ | llm-compressor | perplexity done (1B, 3B, 3 seeds) |

Speed and memory: vLLM (Marlin W4A16 kernels).

## Group B: 8-bit weights and activations (W8A8 INT8)

Goal: faster high-load serving (INT8 math). FP8 is not available on the A40.

| Method | Implementation | Status |
|---|---|---|
| RTN W8A8 (reference) | llm-compressor | planned |
| SmoothQuant | llm-compressor | planned |

Speed and memory: vLLM (INT8 kernels).

## Group C: 4-bit weights, activations and KV cache (W4A4KV4)

Goal: the most aggressive format. Activation outliers must be removed first (rotations).

| Method | Implementation | Status |
|---|---|---|
| RTN W4A4 (reference) | SpinQuant codebase, no rotation | planned |
| QuaRot (official) | Official spcl/QuaRot, ported to Llama-3.2 (its transformers 4.38 pin and model whitelist exclude it) | planned |
| QuaRot (via SpinQuant code) | SpinQuant codebase with random Hadamard rotations and no training: the same algorithm in a different codebase | planned |
| SpinQuant | Official facebookresearch/SpinQuant (transformers 4.44.2, has Llama-3.2 support) | planned |
| DuQuant | Official Hsu1023/DuQuant, ported to a newer transformers version (its pinned 4.35.2 cannot load Llama-3.2) | planned |

**QuaRot comparison (two implementations):** same model, same calibration samples, settings aligned flag by flag. We compare perplexity, zero-shot accuracy, quantization time, the spread over several rotation seeds, and the porting effort. If results differ, we trace the difference to a specific setting or code path.

These codebases simulate quantization (fake quant), so this group reports **accuracy and quantization cost**. vLLM has no W4A4 kernels for the A40, so end-to-end speed is not measured with vLLM. The official QuaRot code includes real CUTLASS INT4 kernels (built for sm_86), so a kernel-level speed test is possible as an extra.

## Fairness rules

1. **Same calibration data:** our 128 × 2048-token C4 windows (`scripts/calib_data.py`), exported to a file for the research codebases. Their default is WikiText-2 train, which would overlap with the evaluation data.
2. **Same perplexity protocol:** WikiText-2 test, 2048-token windows. Each codebase must reproduce our BF16 perplexity (9.756 / 7.817) before we trust its quantized results.
3. **Same zero-shot tasks and lm-eval version** for every method.
4. **Same settings within a group** (bits, group size, symmetric or not, KV bits), listed in the results table.

## Environments (one venv each, because the codebases pin conflicting versions)

| venv | Used for |
|---|---|
| `quant` | llm-compressor: groups A and B (exists) |
| `vllm` | speed and memory for groups A and B |
| `spinquant` | SpinQuant and QuaRot-via-SpinQuant: transformers 4.44.2, an older PyTorch with a CUDA 12.1 build, fast-hadamard-transform compiled |
| `quarot` | Official QuaRot: transformers upgraded to at least 4.43 for Llama-3.2, fast-hadamard-transform and CUTLASS kernels compiled |
| `duquant` | DuQuant |

## Known risks

- **Compiling CUDA code** (fast-hadamard-transform for QuaRot and SpinQuant, CUTLASS kernels for QuaRot): the server's nvcc 12.0 does not support gcc 13.2. Possible fixes: a newer CUDA toolkit installed in the home folder (no sudo needed), or gcc 12. Test this first when starting group C.
- **Official QuaRot needs porting:** transformers 4.38 cannot read Llama-3.2's "llama3" RoPE scaling (added in 4.43), Llama-3.2 is not in its model whitelist, and Llama-3.2's tied embeddings must be untied before rotating (SpinQuant already does this). Its zero-shot code uses an old lm-eval API.
- **DuQuant needs porting:** same transformers problem (pinned 4.35.2).
- **SpinQuant rotation training** is configured for 8 GPUs. On one GPU we use gradient accumulation to keep the same effective batch. Memory needs for the 3B are unknown.

## Order

1. Group A on 1B and 3B: perplexity, then vLLM speed and memory, plus zero-shot tasks. This gives the first comparison table for the meeting.
2. Group B (SmoothQuant), which reuses the same tools.
3. Group C: SpinQuant environment (RTN W4A4, QuaRot-via-SpinQuant, SpinQuant), then official QuaRot, then DuQuant.

## Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-11-09 | Llama-3.2-1B and 3B, base models | 1B–3B as the mentor suggested. The only family that SpinQuant supports officially, with published results to check against. Base models because the papers report perplexity on base models |
| 2026-11-09 | llm-compressor for RTN, GPTQ, AWQ and SmoothQuant | One toolkit means the same calibration pipeline and checkpoint format, so differences come from the algorithms. Its checkpoints run in vLLM with real low-bit kernels. AutoAWQ and AutoGPTQ are discontinued |
| 2026-11-09 | Calibration on C4 (128 × 2048 tokens), not WikiText-2 | Avoid calibrating on the evaluation data. GPTQ paper's setting |
| 2026-11-09 | `W4A16_ASYM` for every W4A16 method | llm-compressor's examples use symmetric for GPTQ and asymmetric for AWQ. One format for both isolates the algorithm. Asymmetric is what both original papers use |
| 2026-11-09 | RTN added as a reference in every group | Shows how much each method improves over plain rounding |
| 2026-11-09 | SmoothQuant+ dropped | Salem's decision. Its official code was never released |
| 2026-11-09 | QuaRot run with two implementations | Salem's decision: compare the official code against the SpinQuant-codebase version |
| 2026-11-09 | Report calibration-based methods as mean ± std over 3 calibration seeds (0, 1, 2) | The seed study showed GPTQ's result can move by 0.29 PPL with the calibration samples alone; one seed can mislead |
