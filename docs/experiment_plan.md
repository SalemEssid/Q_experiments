# Experiment plan

Models: Llama-3.2-1B (fast iteration) and Llama-3.2-3B (main results), base versions, BF16 baseline.

The methods target different formats, so they are compared **within a group**. Comparisons across groups need care (W4A16 saves memory traffic; W8A8 and W4A4 also save arithmetic).

## Group A: weight-only 4-bit (W4A16, group size 128, asymmetric)

Goal: smaller model, faster single-user generation.

| Method | Implementation | Status |
|---|---|---|
| RTN (reference) | llm-compressor | done (1B, 3B): perplexity, zero-shot, speed, memory |
| GPTQ | llm-compressor | done (1B, 3B; 3 seeds): perplexity, zero-shot, speed, memory |
| AWQ | llm-compressor | done (1B, 3B; 3 seeds): perplexity, zero-shot, speed, memory |

Speed and memory: vLLM (Marlin W4A16 kernels).

## Group B: 8-bit weights and activations (W8A8 INT8)

Goal: faster prefill and high-load serving (INT8 math). FP8 is not available on the A40.

Two activation formats, both INT8 symmetric, with the same weights (INT8 symmetric, one scale per output channel):

| Scheme | Activation scale | Why |
|---|---|---|
| `W8A8` (main) | one per token, computed at run time (dynamic) | llm-compressor's standard W8A8 preset and what vLLM serves. A per-token scale already limits the damage of outliers, so RTN may already be close to BF16 |
| `W8A8_STATIC` (second step) | one per layer, fixed from calibration (static) | The setting SmoothQuant was designed for: one fixed scale lets a single outlier channel squeeze every token, so RTN should suffer and smoothing should matter. Cheaper at inference (no scale computation) |

| Method | Implementation | Status |
|---|---|---|
| RTN W8A8 (reference) | llm-compressor `QuantizationModifier` | done (1B, 3B): perplexity, zero-shot, speed, memory |
| SmoothQuant (α = 0.5) + RTN | llm-compressor `SmoothQuantModifier` + `QuantizationModifier` | done (1B, 3B; 3 seeds): perplexity, zero-shot, speed, memory |
| SmoothQuant α = 0.8 (sensitivity check) | same, `--alpha 0.8` | done (1B, 3B; seed 0): perplexity, zero-shot |
| RTN and SmoothQuant with `W8A8_STATIC` | same, `--scheme W8A8_STATIC` | done as a diagnosis: broken on both models (1B ≈ 1300, 3B 60.6), cause found (massive activations in `down_proj`). RTN static speed measured: within 1% of per-token |

Steps: (1) RTN W8A8 on the 1B and check that activations really are quantized in evaluation ✓; (2) SmoothQuant on the 1B ✓; (3) static activations and diagnosis ✓; (4) both models, 3 seeds for SmoothQuant, zero-shot ✓ (`run_groupB.sh`); (5) vLLM speed ✓ (`run_speed_groupB.sh`). **Group B is complete.**

Speed and memory: vLLM (INT8 kernels).

## Group C: 4-bit weights, activations and KV cache (W4A4KV4)

Goal: the most aggressive format. Activation outliers must be removed first (rotations).

| Method | Implementation | Status |
|---|---|---|
| BF16 in each codebase | must reproduce 9.756 / 7.817 | SpinQuant code: done (9.761 / 7.817) |
| RTN W4A4 (reference) | SpinQuant codebase, no rotation | running: `run_groupC_rtn.sh`. GPTQ without rotation: `run_groupC_gptq.sh` |
| QuaRot (official) | Official spcl/QuaRot, ported to Llama-3.2 (its transformers 4.38 pin and model whitelist exclude it) | planned |
| QuaRot (via SpinQuant code) | SpinQuant codebase with random Hadamard rotations and no training: the same algorithm in a different codebase | RTN weights, 3 rotation seeds: running (`run_groupC_rtn.sh`). GPTQ weights on our C4 windows, 3 rotation seeds: `run_groupC_gptq.sh` (task 1) |
| SpinQuant | Official facebookresearch/SpinQuant (transformers 4.44.2, has Llama-3.2 support) | rotation training on our 800 C4 windows (1 GPU, 8 accumulation steps), then GPTQ: `run_spinquant_train.sh` (task 2) |
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
| `spinquant` | SpinQuant and QuaRot-via-SpinQuant: transformers 4.44.2, PyTorch 2.7.1+cu126, fast-hadamard-transform from a prebuilt wheel (`scripts/setup_spinquant_env.sh`) |
| `quarot` | Official QuaRot: transformers upgraded to at least 4.43 for Llama-3.2, fast-hadamard-transform and CUTLASS kernels compiled |
| `duquant` | DuQuant |

## Known risks

- **Compiling CUDA code** (fast-hadamard-transform for QuaRot and SpinQuant, CUTLASS kernels for QuaRot): the server's nvcc 12.0 does not support gcc 13.2. For fast-hadamard-transform this is avoided with a prebuilt wheel (v1.1.0.post2 has wheels for PyTorch 2.6–2.10 with CUDA 12, Python 3.11; all of them need the new C++ ABI, so PyTorch ≥ 2.7). Official QuaRot's own CUTLASS kernels still need compiling. Possible fixes: a newer CUDA toolkit installed in the home folder (no sudo needed), or gcc 12.
- **Official QuaRot needs porting:** transformers 4.38 cannot read Llama-3.2's "llama3" RoPE scaling (added in 4.43), Llama-3.2 is not in its model whitelist, and Llama-3.2's tied embeddings must be untied before rotating (SpinQuant already does this). Its zero-shot code uses an old lm-eval API.
- **DuQuant needs porting:** same transformers problem (pinned 4.35.2).
- **SpinQuant rotation training** is configured for 8 GPUs. On one GPU we use gradient accumulation to keep the same effective batch. Memory needs for the 3B are unknown.
- **No NCCL on the vGPU** (`cudaMemPoolCreate` not supported): SpinQuant runs with the gloo backend (our patch). Rotation training must therefore run without FSDP. If Hugging Face's Trainer or accelerate tries to start NCCL itself, that needs a fix too.

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
| 2026-11-10 | Zero-shot batch size 16 for every model | Batch 32 ran out of memory on the 3B. Batch size shifts the 1B averages by −0.26 to +0.16 points, so it must be the same for all models |
| 2026-11-10 | SmoothQuant rounds the weights with RTN, not GPTQ (llm-compressor's example uses GPTQ) | The paper uses RTN. With RTN, the difference to the RTN W8A8 reference comes from smoothing alone. 8-bit weights lose little with RTN, so GPTQ would add little |
| 2026-11-10 | SmoothQuant α = 0.5 as the main setting, α = 0.8 as a sensitivity check | 0.5 is the paper's and llm-compressor's default, fixed before seeing results, so α is not tuned on the test set. 0.8 is the value in llm-compressor's Llama W8A8 example |
| 2026-11-10 | Two activation formats: dynamic per-token (main) and static per-tensor | With per-token scales RTN may already be near lossless, which would hide what SmoothQuant does. The static format is the case it was built for |
| 2026-11-10 | RTN W8A8 (dynamic) runs once, not with 3 seeds | It uses no calibration data: weight scales come from the weights, activation scales from each token at run time |
| 2026-11-10 | Group B main results use per-token (dynamic) activation scales. Static scales are reported as a diagnosis, without seeds or zero-shot | Static scales break Llama-3.2 (PPL ≈ 1300) with or without SmoothQuant, because of massive activations in `down_proj`'s input. Running seeds and tasks on a broken model would add nothing |
| 2026-11-10 | SpinQuant environment: PyTorch 2.7.1+cu126 with SpinQuant's pinned transformers 4.44.2, and fast-hadamard-transform from a prebuilt wheel | No compiling, so the nvcc 12.0 / gcc 13.2 problem is avoided. 2.7 is the oldest PyTorch whose C++ ABI matches the prebuilt 1.1 wheels (2.6 failed: its wheels use the old ABI, and the wheel labelled for it is in fact built with the new one). Keep transformers at SpinQuant's pin, because its code contains its own copy of the 4.44 Llama model. If 2.7 clashes with transformers 4.44.2, the fallback is to compile fast-hadamard-transform for an older PyTorch with a home-folder CUDA toolkit |
| 2026-11-10 | SpinQuant code pinned to commit `8f47aa3` | Reproducibility: the results must be traceable to the exact code |
| 2026-11-10 | Changes to official code only through `scripts/patch_spinquant.sh`, each one justified in experimental_setup.md. First change: gloo instead of NCCL | NCCL cannot start on the vGPU. Keeping every change in one script makes the difference to the official code visible (`git diff`) and repeatable |
| 2026-11-10 | Group C settings for every method: weights 4-bit symmetric per output channel with clipping search; activations 4-bit asymmetric per token (lm_head BF16); KV cache 4-bit asymmetric, one scale per token and head | The settings of SpinQuant's own scripts (`--w_clip --a_asym --k_asym --v_asym`), which reports Llama results. KV group = head_dim (64 on the 1B, 128 on the 3B) instead of the scripts' 128, because SpinQuant only accepts head_dim or per-token groups for the keys, and values always use head_dim. The official QuaRot and DuQuant runs will be set to match these flags |
| 2026-11-10 | Group C differences are measured against the BF16 result of the same codebase | Removes small evaluation offsets between codebases (SpinQuant's BF16 is 9.761 vs our 9.756 on the 1B) |
| 2026-11-10 | RTN weights first (BF16, RTN without rotation, QuaRot-RTN with 3 rotation seeds), GPTQ weights after | RTN needs no calibration data, so these runs work before SpinQuant's calibration loader is switched from WikiText-2 to our C4 samples. They show what the rotation alone does |
