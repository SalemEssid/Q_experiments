# Methods notes

Working notes for the final report: how each method works, how we run it, and what we observe.
Related files: [experiment_plan.md](experiment_plan.md) (plan and decisions), [experimental_setup.md](experimental_setup.md) (hardware and software), [results.md](results.md) (numbers).

Paper summaries below are our reading of the papers; check exact claims against the papers before quoting them in the report.

---

## 1. Background

### Notation
**WxAy(KVz):** weights in x bits, activations in y bits, KV cache in z bits. Example: W4A16 = 4-bit weights, 16-bit activations.

- **Weight-only (W4A16):** weights are stored in 4 bits and expanded back to BF16 just before each matrix multiply. This saves memory and memory traffic, but the math is still BF16.
- **Weight and activation (W8A8, W4A4):** both inputs of the matrix multiply are low-bit, so the GPU can use integer math (INT8 or INT4 tensor cores). This also saves arithmetic.

### Uniform integer quantization
A group of values (here 128 consecutive weights) shares one **scale** and, for asymmetric quantization, one **zero-point**:

- quantize: q = clamp(round(w / scale) + zero_point, 0, 2^b − 1)
- dequantize: ŵ = (q − zero_point) × scale
- asymmetric: scale = (max − min) / (2^b − 1), so the 16 levels cover exactly the group's range
- symmetric: zero_point is fixed, and the range is ±max|w|

Smaller groups give a more precise scale but more storage. Group size 128 is the standard trade-off for 4-bit weights.

### Why activations are harder than weights
Weights have smooth, narrow distributions. Activations of LLMs contain **outlier channels**: a few hidden dimensions whose values are 10–100× larger than the rest, consistently across tokens. With one scale per token, the outliers stretch the range, and all the normal values get rounded to just a few levels. 8-bit can tolerate this with tricks (SmoothQuant); 4-bit cannot without removing the outliers first (QuaRot, SpinQuant, DuQuant).

### Two families of methods
1. **Better rounding of the weights:** GPTQ chooses the rounding of each weight so that the layer's output changes as little as possible.
2. **Transforming the model before rounding:** rewrite the model into a mathematically equivalent one that is easier to quantize.
   - Per-channel **scaling** (AWQ, SmoothQuant): multiply one side by s and the other by 1/s.
   - **Rotations** (QuaRot, SpinQuant, DuQuant): multiply by an orthogonal matrix that spreads outliers over all channels.

The two families combine. QuaRot and SpinQuant rotate first, then round with GPTQ.

---

## 2. Overview

| Method | Paper | Target format | Core idea | Calibration data | Training | Extra work at inference | Weight rounding |
|---|---|---|---|---|---|---|---|
| RTN | (baseline) | any | Round each weight to the nearest level | none | no | none | RTN |
| GPTQ | Frantar et al., ICLR 2023 | W4A16, W3A16 | Round weights one column at a time, correcting the error on the remaining columns | yes | no | none | GPTQ |
| AWQ | Lin et al., MLSys 2024 | W4A16 | Scale up the weight channels that see large activations before rounding | yes (small) | no (grid search) | none (scales folded into previous layer) | RTN |
| SmoothQuant | Xiao et al., ICML 2023 | W8A8 | Move outliers from activations to weights with per-channel scales | yes | no | none (folded); enables INT8 math | RTN in the paper (llm-compressor's example uses GPTQ) |
| QuaRot | Ashkboos et al., NeurIPS 2024 | W4A4KV4 | Random Hadamard rotations remove outliers; most are folded into the weights | for GPTQ only | no | online Hadamard transforms (before `down_proj`, on queries/keys) | GPTQ or RTN |
| SpinQuant | Liu et al., ICLR 2025 | W4A4KV4, W4A8, W4A16 | Learn the rotations instead of choosing them at random | yes (rotation training + GPTQ) | rotations only (~100 steps) | same as QuaRot | GPTQ or RTN |
| DuQuant | Lin et al., NeurIPS 2024 | W4A4, W6A6 | Block-wise rotations built from known outlier positions, plus a zigzag permutation, plus smoothing | yes | no | online block-wise rotations and permutation | RTN with clipping |

---

## 3. RTN (round-to-nearest): reference

**Idea:** quantize each group of weights independently with the formula above. No data, no optimization.

**Why we include it:** every paper uses it as the lower bound. The useful number for a method is how much of RTN's accuracy loss it recovers:
recovered = (PPL_RTN − PPL_method) / (PPL_RTN − PPL_BF16).

**How we run it:** llm-compressor `QuantizationModifier`, scheme `W4A16_ASYM` (group 128), `lm_head` not quantized.

**Observations:**
- Llama-3.2-1B: PPL 9.756 → 11.709 (+20%). Takes 1.9 s and almost no GPU memory. The sample generation is still fluent: a 20% perplexity increase does not make the model unusable, but it does measurably hurt accuracy.
- Llama-3.2-3B: PPL 7.817 → 8.490 (+8.6%), less than half the 1B's relative loss. Larger models tolerate rounding better.
- **W8A8 (group B):** 1B PPL 9.756 → 9.888 (+1.4%) in 1.6 s, with no calibration data. 8 bits give 256 levels instead of 16, so plain rounding is already good when the activation scale is set per token.
- Zero-shot: 1B 60.44 → 55.12 (−5.3 points), the largest loss in group A. About half of it is BoolQ, which falls to 48.3%: below a coin flip and below answering "yes" to everything (62.2%). 3B: 68.89 → 67.44 (−1.45 points).

---

## 4. GPTQ

**Paper:** Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023.

**Idea:** quantize one layer at a time and choose the rounding so that the layer's **output** on real inputs changes as little as possible:
minimize ‖W X − Ŵ X‖², where X are the layer's inputs on calibration data.

**How it works:**
1. Run calibration data through the model and collect each layer's inputs X. The matrix H = X Xᵀ describes which input directions matter (it is the Hessian of the objective).
2. Quantize the weight matrix one column at a time. After rounding a column, update all columns not yet quantized to compensate for the rounding error, using H⁻¹. Later columns absorb the error of earlier ones.
3. Numerical tricks that make it fast and stable: process columns in blocks of 128 (lazy batch updates), use a Cholesky decomposition of H⁻¹, add a small dampening term to H's diagonal (1% of its mean).
4. Optional **act-order:** quantize the columns with the largest activations first, so the most important ones are rounded while there is the most freedom left to compensate.
5. Go layer by layer, feeding the already-quantized outputs into the next layer, so later layers also compensate for earlier errors.

**Calibration in the paper:** 128 random 2048-token segments of C4. This is the setting we use for all methods.

**How we run it:** llm-compressor 0.15.0 `GPTQModifier`, scheme `W4A16_ASYM` (group 128), `lm_head` not quantized. Settings recorded in `recipe.yaml`:
- block size 128, dampening 1% (`dampening_frac: 0.01`)
- act-order `static`. In llm-compressor this is an alias for `weight`: columns are processed in order of activation importance during calibration only, and the saved weights keep the normal layout, so inference costs nothing extra. The older `group` variant, which needs an index at inference, has been removed.
- Act-order is a later addition from the GPTQ code, not part of the core algorithm in the paper (check before stating). Our GPTQ is therefore probably somewhat stronger than the paper's basic version.

**Observations:**
- Llama-3.2-1B: PPL 10.610 (+8.7%). Recovers 56% of RTN's loss, the best of the three W4A16 methods. Quantization took 72 s, with a peak GPU memory of 2.3 GiB: llm-compressor moves one layer at a time to the GPU.
- Llama-3.2-3B: PPL 8.241 (+5.4%). Recovers 37% of RTN's loss. 167 s, 2.7 GiB of GPU.
- **Sensitive to the calibration samples.** Over 3 seeds: 1B 10.699 ± 0.164 (range 10.598–10.889), 3B 8.254 ± 0.011. This is expected from how GPTQ works: it solves a least-squares problem on the exact calibration inputs (it fits H = X Xᵀ), so it partly fits the particular samples. With an unlucky set (seed 1 on the 1B) it loses about a third of its improvement over RTN.
- **Zero-shot (3 seeds):** 1B 58.76 ± 0.37 (−1.68 points vs BF16, recovers 68% of RTN's loss); 3B 67.75 ± 0.31 (−1.14, recovers only 21%). On the 3B it is barely better than RTN (+0.31 points), although it has the best perplexity. **Its perplexity advantage does not carry over to the tasks.** Possible reason: it fits the layer outputs on the calibration text, which helps on similar text (what perplexity measures) more than on question answering.

---

## 5. AWQ

**Paper:** Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan, Han. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, MLSys 2024 (best paper).

**Idea:** a small fraction of weights (about 0.1–1%) matter much more than the rest. They are the weights in input channels that receive **large activations**, not the weights that are large themselves. Keeping them in FP16 would help, but mixed precision is hard to run efficiently. Instead, AWQ scales them up before rounding.

**How it works:**
1. For each input channel, measure the average activation size s_X on calibration data.
2. Replace W·x with (W·diag(s))·(diag(s)⁻¹·x), which gives the same result. Multiplying important channels by s > 1 before rounding reduces their *relative* rounding error.
3. Choose s = s_X^α, with α found by a grid search (20 values in [0, 1]). The objective is the quantized block's output error on calibration data. The factor 1/s is folded into the previous operation (RMSNorm or the previous linear layer), so inference costs nothing extra.
4. Optionally search a clipping range for the weights.
5. Round with plain RTN. AWQ does not correct rounding errors; it only makes the important weights less sensitive to them.

**Paper claims:** no backpropagation or reconstruction, so it needs little calibration data (128 × 512 tokens from the Pile) and is less likely to overfit it. Quantization is cheaper than GPTQ.

**How we run it:** llm-compressor 0.15.0: `AWQModifier(duo_scaling="both")` followed by `QuantizationModifier` with `W4A16_ASYM`. Details from `recipe.yaml` and the 0.15.0 source code:
- **Four smoothing points per layer:** `input_layernorm` → q/k/v_proj; `v_proj` → `o_proj`; `post_attention_layernorm` → gate/up_proj; `up_proj` → `down_proj`.
- **The `v_proj` → `o_proj` point is skipped on Llama-3.2.** Grouped-query attention makes `v_proj`'s output (8 key/value heads) smaller than `o_proj`'s input (all query heads), so the scales cannot be matched. llm-compressor skips it and logs a warning.
- **Grid search:** 20 candidates per smoothing point (`n_grid: 20`). With `duo_scaling="both"`, these are 10 values of α × 2 formulas: s = s_X^α (activations only) and s = s_X^α / s_W^(1−α) (activations and weights). It keeps the scale with the lowest MSE between the block's quantized and original outputs. "both" does **not** make the search longer.
- **No weight-clipping search.** The official AWQ code (llm-awq) and AutoAWQ also search, for each layer, a clipping range that cuts off the most extreme weights before rounding. llm-compressor's AWQ has no such step: the word "clip" does not appear in its source. **Our "AWQ" is therefore the scaling step only,** which must be stated in the report. It may explain part of the gap with GPTQ.

**Observations:**
- Llama-3.2-1B: PPL 10.906 (+11.8%). Recovers 41% of RTN's loss, behind GPTQ. Quantization took 215 s and 9.2 GiB of GPU: 3× slower and 4× more memory than GPTQ, the **opposite** of the paper's cost claim.
- Llama-3.2-3B: PPL 8.282 (+6.0%). Recovers 31%. 572 s, 11.6 GiB. The gap to GPTQ shrinks to 0.04 PPL.
- **Very stable across calibration samples,** which confirms the paper's robustness claim. Over 3 seeds: 1B 10.907 ± 0.005 (about 30× less spread than GPTQ), 3B 8.280 ± 0.003 (about 4× less). AWQ only uses average activation magnitudes per channel to pick its scales, and these statistics barely change between sample sets.
- **Zero-shot (3 seeds): the best W4A16 method.** 1B 58.99 ± 0.45 (−1.45 points, recovers 73%; a tie with GPTQ). 3B 68.68 ± 0.33 (−0.21, recovers 86%): almost lossless, and clearly ahead of GPTQ (every AWQ seed beats every GPTQ seed). The gain over GPTQ comes from ARC-e, ARC-c and BoolQ.
- Its zero-shot seed std (0.33–0.45) is as large as GPTQ's, even though its perplexity barely moves between seeds. The zero-shot average has its own noise: tiny weight changes flip near-tie questions (mostly BoolQ).
- **Trade-off:** GPTQ has slightly better perplexity. AWQ has better zero-shot accuracy (clearly on the 3B) and much more stable perplexity, but costs 3× more time and 4× more GPU memory to quantize in llm-compressor.
- Why AWQ costs more here: it runs each block once per candidate scale (20 times) on all 262k calibration tokens and keeps the block inputs on the GPU. GPTQ needs one pass per layer to collect statistics. Our 2048-token samples (4× the tokens of AWQ's default 128 × 512) make this heavier.

**Points for the report:**
- Seed study answered: GPTQ is ahead in perplexity in every seed but by a negligible 0.027 PPL on the 3B, and AWQ is far more stable across calibration sets.
- Zero-shot answered: **the perplexity ranking does not hold.** AWQ ties GPTQ on the 1B and beats it by 0.9 points on the 3B (results.md, "Accuracy: zero-shot tasks").
- Possible extra: perplexity on C4, the calibration data, to test whether GPTQ's perplexity lead comes from fitting the calibration text.
- Possible extras: AWQ with the paper's calibration size (128 × 512) to test the cost claim, and the original AWQ code (with clipping) to measure what the missing step is worth.

---

## 6. SmoothQuant (in progress)

**Paper:** Xiao, Lin, Seznec, Wu, Demouth, Han. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*, ICML 2023.

**Idea:** for W8A8, the activations are hard to quantize (outlier channels), while the weights are easy. Move part of the difficulty from activations to weights with a per-channel scale:
Y = (X · diag(s)⁻¹) · (diag(s) · W), with s_j = max|X_j|^α / max|W_j|^(1−α).

**How it works:**
1. Measure the maximum activation per channel on calibration data.
2. Compute s with the migration strength α (0.5 in the paper, larger for models with stronger outliers).
3. Fold diag(s)⁻¹ into the previous RMSNorm and diag(s) into the weights. The model is exactly equivalent before quantization.
4. Quantize weights per channel and activations per token to INT8. The matrix multiply then runs on INT8 tensor cores.

**Paper's quantization levels (check the exact table before quoting):** O1 to O3 go from finer to coarser activation scales: per-token dynamic, per-tensor dynamic, per-tensor static. Coarser is faster but harder. The paper's main point is that smoothing makes even the coarse static setting work.

**How we run it:** llm-compressor 0.15.0 `SmoothQuantModifier(smoothing_strength=α)` followed by `QuantizationModifier` (RTN, as in the paper). Details from the 0.15.0 source code:
- **Two smoothing points per layer on Llama:** `input_layernorm` → q/k/v_proj and `post_attention_layernorm` → gate/up_proj. The scale is folded into the RMSNorm weight. The inputs of `o_proj` and `down_proj` are **not** smoothed, because no norm sits directly before them. On Llama, `down_proj`'s input is known to have the largest outliers, so this is a limitation to keep in mind when reading the results.
- **Activation statistic:** the largest absolute value of each channel over all calibration tokens (a running min/max hook on the norm output).
- **Scale formula:** s_j = max|X_j|^α / (2 · max|W_j|)^(1−α), with the weight maximum taken over all layers that share the input (q, k and v together). The factor 2 multiplies every channel's scale by the same constant, which changes neither per-token nor per-channel INT8 rounding.
- **α = 0.5** (paper and llm-compressor default) as the main setting, decided in advance. **α = 0.8** (llm-compressor's Llama W8A8 example) as a sensitivity check.
- **Formats:**
  - `W8A8` preset: INT8 symmetric weights with one scale per output channel, INT8 symmetric activations with one scale per token computed at run time. This is finer than any of the paper's levels.
  - `W8A8_STATIC` (our custom scheme): the same weights, with one fixed activation scale per layer measured on the calibration data (llm-compressor's default `minmax` observer, a moving average over calibration batches). This is close to the paper's O3, except that our weights are per channel.
- llm-compressor's own W8A8 example rounds the weights with GPTQ instead. We use RTN so that the only difference to the RTN reference is the smoothing (decision log, 2026-11-10).

**How accuracy is measured for W8A8:** when Hugging Face loads the checkpoint, compressed-tensors replaces each quantized Linear layer's forward function with one that rounds the input activations to INT8 before the multiply (fake quantization; `set_forward_quantized` in compressed-tensors 0.20.1). `ppl_eval.py` proves this on every run: it scores the first window with activation quantization on and off, and the two losses must differ. "Off" keeps the 8-bit weights, so it also separates the weight error from the activation error.

---

## 7. QuaRot (planned, two implementations)

**Paper:** Ashkboos, Mohtashami, Croci, Li, Cameron, Jaggi, Alistarh, Hoefler, Hensman. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*, NeurIPS 2024.

**Idea:** multiplying the hidden states by an orthogonal matrix Q (QQᵀ = I) does not change what the model computes if the weights are multiplied by Qᵀ. A **random Hadamard matrix** spreads each outlier over all channels, so after rotation no channel is extreme and 4-bit activations become possible.

**How it works:**
1. **Computational invariance (from SliceGPT):** fold each RMSNorm's scale into the next linear layer. RMSNorm then commutes with rotations, so the whole residual stream can be rotated. Each weight matrix that reads from the residual stream is multiplied by Q, and each one that writes to it by Qᵀ. This is done offline with no cost at inference.
2. **Online Hadamard transforms** where folding is impossible: before `down_proj` (the input comes from the SwiGLU product, not the residual stream), and on queries and keys after RoPE (for KV-cache quantization). The fast Walsh–Hadamard transform costs O(d log d).
3. **Attention values/outputs:** a per-head Hadamard folded into `v_proj` and `o_proj`.
4. Quantize weights with GPTQ (or RTN), activations per token to 4 bits, and the KV cache to 4 bits (group 128).

**Two implementations we compare:**
- **Official (spcl/QuaRot):** needs porting to Llama-3.2 (transformers version, model whitelist, tied embeddings). Includes real INT4 CUTLASS kernels.
- **Inside SpinQuant's codebase:** random Hadamard rotations, no rotation training. This is the same algorithm. SpinQuant's paper uses it as its baseline.

**Expected differences to watch for:** default settings (symmetric vs asymmetric activations and KV cache, KV group size, weight clipping), how the random Hadamard is generated, GPTQ details (act-order, dampening), calibration loaders, and evaluation code. We align the settings flag by flag and use the same calibration samples. Remaining differences point to implementation details.
**Rotation seed:** SpinQuant's paper reports that different random rotations give quite different accuracy, so we run several seeds and report the spread.

---

## 8. SpinQuant (planned)

**Paper:** Liu, Zhao, et al. (Meta). *SpinQuant: LLM Quantization with Learned Rotations*, ICLR 2025.

**Idea:** QuaRot's rotations are random, and the accuracy after quantization varies a lot from one random rotation to another. So **learn** the rotation that gives the best quantized model.

**How it works:**
1. Four rotations: **R1** on the residual stream and **R2** per attention head (values/outputs), both folded into the weights; **R3** (queries/keys, for the KV cache) and **R4** (before `down_proj`) are fixed online Hadamards, as in QuaRot.
2. Learn R1 and R2 by minimizing the quantized model's next-token loss on calibration data. Activations are quantized during training and weights stay in 16 bits. Gradients pass through the rounding with the straight-through estimator.
3. To keep the matrices orthogonal during training, use **Cayley SGD** (optimization on the Stiefel manifold). Only the rotations are trained; the model weights stay frozen. The authors' scripts use 800 WikiText-2 samples and 100 steps.
4. Fold the learned rotations into the weights, then quantize the weights with GPTQ.
5. Variants: with online Hadamards (R3, R4), or without them so the model keeps the standard architecture.

**How we will run it:** official facebookresearch/SpinQuant, which handles Llama-3.2's tied embeddings. One GPU with gradient accumulation instead of the authors' 8 GPUs. Rotation training on our C4 samples, not WikiText-2.

---

## 9. DuQuant (planned)

**Paper:** Lin, Xu, et al. *DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs*, NeurIPS 2024.

**Idea:** besides normal outlier channels, LLMs have **massive outliers**: a few extremely large values in specific tokens and channels, especially at the input of `down_proj`. A random rotation spreads them only partly. DuQuant builds rotations that target the known outlier positions.

**How it works:**
1. **Smoothing** (SmoothQuant-style, α ≈ 0.6) to balance activations and weights.
2. **Block-diagonal rotation:** split channels into blocks of 128. In each block, build a rotation greedily from the measured activations: repeatedly find the largest outlier dimension and rotate to spread it (up to 256 greedy steps).
3. **Zigzag permutation:** reorder channels so that each block gets a similar total outlier size, then rotate again. That gives two rotations with a permutation between them, the "dual transformation".
4. Quantize weights with RTN plus clipping (the authors' script uses clipping ratios of 0.8 for weights and 0.9 for activations). The KV cache uses the activation bit-width.

**How we will run it:** official Hsu1023/DuQuant, ported to a newer transformers version for Llama-3.2. Calibration on our C4 samples (their default is WikiText-2).

---

## 10. Implementation notes (for the methodology section)

- llm-compressor 0.15.0 moves one layer at a time to the GPU during calibration, so quantization peak GPU memory is far below the model size for RTN and GPTQ.
- Hugging Face transformers runs the quantized checkpoints without fast low-bit kernels. Evaluation time is the same as BF16 (about 11 s for the 1B). Speed must be measured in vLLM.
- Model size: Llama-3.2-1B and 3B share one matrix between the embedding and `lm_head`, which stays in BF16. In the 1B it is about half of the 4-bit checkpoint (0.49 of 0.96 GiB), so compression is 2.4×, not 4×.
- The research codebases (QuaRot, SpinQuant, DuQuant) simulate quantization: values are rounded, but the math stays in 16 bits. Accuracy is valid; speed is not.
- vLLM runs our W4A16 checkpoints with the **Marlin** kernel: it reads the packed 4-bit weights, unpacks them to BF16 in fast on-chip memory and multiplies in BF16. Greedy output is word-for-word the same as Hugging Face's, so the speed and accuracy numbers describe the same model.

## 10b. What the W4A16 speed results teach (for the discussion section)

- **Weight-only quantization speeds up memory-bound work only.** Single-request generation became 1.91× (1B) and 2.34× (3B) faster, about 82% of the speedup predicted from the byte ratio. Prefill and high-load throughput did not improve (0.93–1.02×), because they are limited by arithmetic, which W4A16 still does in BF16.
- **This is exactly the distinction the mentor pointed out between W4A16 and W8A8.** W8A8 (SmoothQuant) and W4A4 (QuaRot, SpinQuant, DuQuant) also quantize activations so the INT8/INT4 tensor cores can be used. That is the only way to speed up the compute-bound workloads.
- **The speed of a W4A16 model depends only on its format,** not on the method that chose the rounding (RTN, GPTQ and AWQ were within 1%). The choice of method is therefore purely about accuracy and quantization cost.
- **For a fixed budget, quantize a bigger model:** a 4-bit 3B matches a BF16 1B in speed (209 vs 222 tok/s) and memory (2.15 vs 2.32 GiB), with much better perplexity (8.25 vs 9.76) and zero-shot accuracy (68.7 vs 60.4 with AWQ).

## 10c. What the zero-shot results teach (for the discussion section)

- **Perplexity and task accuracy can rank methods differently.** GPTQ has the best perplexity on both models; AWQ has the best zero-shot accuracy on the 3B by 0.9 points, where GPTQ is barely better than RTN. Do not rank methods on perplexity alone. This is why the mentor asked for both.
- **Hypothesis to check:** GPTQ fits the calibration data, which helps on similar text. Test: perplexity on C4 (the calibration source). If GPTQ's lead over AWQ grows there, it supports the hypothesis.
- **Zero-shot has a noise floor of about 0.3–0.5 points** in the 6-task average from calibration seeds alone, and about 0.2 points from the batch size. Smaller differences between methods mean nothing for a single checkpoint. Several seeds and the same evaluation settings for every model are needed.
- **On small models, BoolQ mostly measures a yes/no bias.** The BF16 1B is only 1.5 points above answering "yes" to everything, and RTN falls 14 points below it. One badly shifted task can dominate the average, so per-task numbers must be reported too.
- **Size matters more than the method:** the 3B RTN (67.4) beats every 1B model, including BF16 (60.4).

## 11. Questions for the mentor

- llm-compressor's AWQ has no weight-clipping search, and it was slower and used more memory than GPTQ, the opposite of the AWQ paper's claim. Should we also run the original AWQ code (with clipping), or re-run with the paper's calibration size (128 × 512 tokens)?
- Perplexity and zero-shot disagree: GPTQ has the better perplexity, AWQ the better zero-shot accuracy (0.9 points on the 3B). Which should weigh more in our conclusions? Is it worth testing the "GPTQ fits the calibration data" idea with perplexity on C4, or adding a harder task such as MMLU?
- The zero-shot average varies by 0.3–0.5 points between calibration seeds. Are 3 seeds enough, and should groups B and C also use 3 seeds?
- Group C will have no end-to-end speed numbers (no W4A4 kernels in vLLM for the A40). Is accuracy plus kernel-level speed (QuaRot's CUTLASS kernels) enough?
