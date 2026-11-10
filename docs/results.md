# Results

Hardware, software and measurement protocols: see [experimental_setup.md](experimental_setup.md).

**Perplexity protocol:** WikiText-2 (raw) test split, joined and tokenized once, non-overlapping 2048-token windows, PPL = exp(mean next-token NLL). This is the same protocol as the GPTQ, AWQ, QuaRot and SpinQuant papers. Script: `scripts/ppl_eval.py`.
**Speed protocol:** vLLM 0.32.0, same settings for every model (`scripts/bench_vllm.py`).
**Regenerate all tables:** `python scripts/summarize.py` (uses full-precision values).

## Main comparison table: group A (W4A16, group size 128, asymmetric)

"Format" is WxAy: weights in x bits, activations in y bits. W4A16 and W8A8 methods solve different problems (memory vs. compute), so compare across groups with care.

| Model | Method | Format | WikiText-2 PPL ↓ | Zero-shot avg ↑ | Size on disk | Weights in GPU | KV cache (tokens) | Decode tok/s, 1 request ↑ | Throughput, 256 requests (out tok/s) ↑ | Quantization cost |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | BF16 baseline | W16A16 | 9.756 | 60.44 | 2.30 GiB* | 2.32 GiB | 540k | 222 | 8,755 | – |
| Llama-3.2-1B | RTN (reference) | W4A16 | 11.709 | 55.12 | 0.96 GiB | 0.99 GiB | 568k | 423 (1.91×) | 8,616 (0.98×) | 1.9 s, 0.06 GiB GPU |
| Llama-3.2-1B | GPTQ | W4A16 | **10.699 ± 0.164** | 58.76 ± 0.37 | 0.96 GiB | 0.99 GiB | 570k | 425 (1.91×) | 8,673 (0.99×) | 72 s, 2.3 GiB GPU |
| Llama-3.2-1B | AWQ | W4A16 | 10.907 ± 0.005 | **58.99 ± 0.45** | 0.96 GiB | 0.99 GiB | 568k | 425 (1.91×) | 8,633 (0.99×) | 215 s, 9.2 GiB GPU |
| Llama-3.2-3B | BF16 baseline | W16A16 | 7.817 | 68.89 | 5.98 GiB* | 6.04 GiB | 113k | 89 | 2,535 | – |
| Llama-3.2-3B | RTN (reference) | W4A16 | 8.490 | 67.44 | 2.10 GiB | 2.15 GiB | 144k | 209 (2.34×) | 2,586 (1.02×) | 2.9 s, 0.13 GiB GPU |
| Llama-3.2-3B | GPTQ | W4A16 | **8.254 ± 0.011** | 67.75 ± 0.31 | 2.10 GiB | 2.15 GiB | 144k | 209 (2.34×) | 2,592 (1.02×) | 167 s, 2.7 GiB GPU |
| Llama-3.2-3B | AWQ | W4A16 | 8.280 ± 0.003 | **68.68 ± 0.33** | 2.10 GiB | 2.15 GiB | 144k | 209 (2.34×) | 2,592 (1.02×) | 572 s, 11.6 GiB GPU |

- PPL and zero-shot for GPTQ and AWQ: mean ± sample std over 3 calibration seeds. RTN and BF16 use no calibration data. Speed, memory and cost: seed-0 checkpoints (identical format for every seed).
- Zero-shot avg: mean accuracy (%) over 6 tasks, see "Accuracy: zero-shot tasks". Bold: best 4-bit method on each metric. **The two accuracy metrics disagree:** GPTQ has the best perplexity, AWQ the best zero-shot accuracy.
- Speedups in brackets are relative to BF16 of the same size.
- \* BF16 size on disk = parameter count × 2 bytes. Quantized sizes are the measured `.safetensors` files.
- Weights in GPU and KV cache: from vLLM at the same memory budget (85% of the GPU). The KV cache is the memory left for the tokens of running requests.
- Quantization cost: time and peak GPU memory of the algorithm itself (not loading or saving). Peak CPU RAM was 3.4 GiB (RTN) to 8.9 GiB (AWQ) on the 1B.

## Model size

Both models share one matrix between the embedding and `lm_head` (128,256 × hidden size), and it stays in BF16.

| Model | Shared embedding (BF16) | Share of the 4-bit checkpoint | Compression vs BF16 |
|---|---|---|---|
| Llama-3.2-1B | 0.49 GiB | 51% | 2.40× |
| Llama-3.2-3B | 0.73 GiB | 35% | 2.85× |

## Accuracy: perplexity

### Seed 0 only

| Model | RTN Δ PPL | GPTQ Δ PPL | AWQ Δ PPL | GPTQ recovers | AWQ recovers | GPTQ–AWQ gap |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | +1.953 (+20.0%) | +0.854 (+8.7%) | +1.150 (+11.8%) | 56% | 41% | 0.296 |
| Llama-3.2-3B | +0.673 (+8.6%) | +0.424 (+5.4%) | +0.465 (+6.0%) | 37% | 31% | 0.041 |

"Recovers" = share of RTN's perplexity loss that the method wins back.

### Calibration-seed study

Same method and settings, three different random sets of 128 × 2048 C4 calibration samples (seeds 0, 1, 2). Std is the sample standard deviation over the three seeds (values from `summarize.py`).

| Model | Method | Seed 0 | Seed 1 | Seed 2 | Mean ± std | Spread (max − min) | RTN loss recovered (mean) |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | GPTQ | 10.610 | 10.889 | 10.598 | **10.699 ± 0.164** | 0.291 | 52% |
| Llama-3.2-1B | AWQ | 10.906 | 10.913 | 10.903 | **10.907 ± 0.005** | 0.010 | 41% |
| Llama-3.2-3B | GPTQ | 8.241 | 8.260 | 8.261 | **8.254 ± 0.011** | 0.020 | 35% |
| Llama-3.2-3B | AWQ | 8.282 | 8.282 | 8.278 | **8.280 ± 0.003** | 0.005 | 31% |

Quantization time and memory are the same for every seed (within 0.5 s). The 3B AWQ seeds 0 and 1 differ only in the 4th decimal (8.2822 vs 8.2817).

### Findings (perplexity)

- **Same ranking on both models: GPTQ < AWQ < RTN** (lower is better). GPTQ is ahead in every seed. A likely reason: GPTQ corrects rounding errors (it updates the remaining weights after each column), while AWQ only rescales important channels before plain rounding. Also, llm-compressor's AWQ has **no weight-clipping search**, which the official AWQ code includes (see methods_notes.md).
- **The GPTQ–AWQ gap almost disappears on the 3B:** 0.21 PPL on the 1B, 0.027 (0.3%) on the 3B, a tie in practice.
- **AWQ is 4–30× more stable across calibration sets** (spread 0.010 vs 0.291 on the 1B, 0.005 vs 0.020 on the 3B). This supports the AWQ paper's claim that it depends less on the calibration data. GPTQ fits the layer outputs on the exact calibration samples, so an unlucky set (seed 1 on the 1B) costs it about a third of its improvement over RTN. AWQ only uses average activation sizes, which barely change between sample sets.
- **A single seed is not enough:** seed 0 was a lucky seed for GPTQ on the 1B.
- **The larger model is more robust:** RTN's loss drops from +20.0% (1B) to +8.6% (3B), GPTQ's from +8.7% to +5.4%.
- **The recovered share is lower on the 3B** (35% / 31% vs 52% / 41%). When RTN already does well, much of the remaining error is the inherent cost of 4 bits, which no rounding method removes.

## Accuracy: zero-shot tasks

Six multiple-choice tasks, 0-shot, lm-eval 0.4.13, batch size 16 for every model (protocol in experimental_setup.md). Accuracy in %: `acc_norm` for ARC-e, ARC-c, HellaSwag and PIQA, `acc` for WinoGrande and BoolQ. lm-eval's standard error per task is about 0.5 (HellaSwag, 10k questions) to 1.4 points (ARC-c, 1.2k questions).

### Perplexity vs zero-shot

Both columns of "RTN loss recovered" use the same definition: the share of RTN's loss (vs BF16) that the method wins back. GPTQ and AWQ: means over 3 seeds.

| Model | Method | PPL Δ vs BF16 | PPL: RTN loss recovered | Zero-shot Δ vs BF16 (points) | Zero-shot: RTN loss recovered |
|---|---|---|---|---|---|
| Llama-3.2-1B | RTN | +1.953 | 0% | −5.32 | 0% |
| Llama-3.2-1B | GPTQ | **+0.943** | **52%** | −1.68 | 68% |
| Llama-3.2-1B | AWQ | +1.151 | 41% | **−1.45** | **73%** |
| Llama-3.2-3B | RTN | +0.673 | 0% | −1.45 | 0% |
| Llama-3.2-3B | GPTQ | **+0.437** | **35%** | −1.14 | 21% |
| Llama-3.2-3B | AWQ | +0.463 | 31% | **−0.21** | **86%** |

### Per task, mean over calibration seeds

GPTQ and AWQ per-task values are means of the three seeds (computed from the 1-decimal values below, so ±0.05).

| Model | Method | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | BoolQ | Avg |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | BF16 | 61.8 | 36.9 | 64.3 | 74.7 | 61.3 | 63.7 | 60.44 |
| Llama-3.2-1B | RTN | 56.3 | 32.7 | 60.2 | 72.0 | 61.2 | 48.3 | 55.12 |
| Llama-3.2-1B | GPTQ | 59.1 | 35.6 | 62.5 | 73.6 | 61.1 | 60.6 | 58.76 ± 0.37 |
| Llama-3.2-1B | AWQ | 61.0 | 36.2 | 62.0 | 74.1 | 61.0 | 59.7 | 58.99 ± 0.45 |
| Llama-3.2-3B | BF16 | 71.9 | 46.1 | 74.1 | 77.8 | 69.8 | 73.7 | 68.89 |
| Llama-3.2-3B | RTN | 71.0 | 45.5 | 72.6 | 77.0 | 68.5 | 70.1 | 67.44 |
| Llama-3.2-3B | GPTQ | 70.8 | 44.7 | 72.7 | 77.5 | 69.0 | 71.8 | 67.75 ± 0.31 |
| Llama-3.2-3B | AWQ | 72.4 | 46.2 | 73.2 | 76.8 | 69.5 | 74.1 | 68.68 ± 0.33 |

### Every run (from `summarize.py`)

| Model | Method | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | BoolQ | Avg |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | BF16 | 61.8 | 36.9 | 64.3 | 74.7 | 61.3 | 63.7 | 60.44 |
| Llama-3.2-1B | RTN | 56.3 | 32.7 | 60.2 | 72.0 | 61.2 | 48.3 | 55.12 |
| Llama-3.2-1B | GPTQ | 58.7 | 36.0 | 62.3 | 73.6 | 60.8 | 60.8 | 58.69 |
| Llama-3.2-1B | GPTQ (seed 1) | 58.8 | 35.7 | 62.3 | 73.6 | 61.6 | 58.5 | 58.42 |
| Llama-3.2-1B | GPTQ (seed 2) | 59.9 | 35.2 | 63.0 | 73.5 | 60.9 | 62.5 | 59.16 |
| Llama-3.2-1B | AWQ | 61.1 | 36.3 | 61.9 | 74.2 | 61.6 | 60.9 | 59.31 |
| Llama-3.2-1B | AWQ (seed 1) | 60.9 | 36.2 | 62.0 | 73.8 | 60.3 | 57.7 | 58.48 |
| Llama-3.2-1B | AWQ (seed 2) | 61.1 | 36.0 | 62.0 | 74.3 | 61.2 | 60.5 | 59.18 |
| Llama-3.2-3B | BF16 | 71.9 | 46.1 | 74.1 | 77.8 | 69.8 | 73.7 | 68.89 |
| Llama-3.2-3B | RTN | 71.0 | 45.5 | 72.6 | 77.0 | 68.5 | 70.1 | 67.44 |
| Llama-3.2-3B | GPTQ | 70.4 | 45.0 | 73.0 | 77.5 | 69.0 | 73.8 | 68.11 |
| Llama-3.2-3B | GPTQ (seed 1) | 71.7 | 45.0 | 72.6 | 77.3 | 68.7 | 70.5 | 67.62 |
| Llama-3.2-3B | GPTQ (seed 2) | 70.4 | 44.0 | 72.6 | 77.6 | 69.4 | 71.2 | 67.53 |
| Llama-3.2-3B | AWQ | 73.1 | 47.2 | 73.1 | 76.8 | 69.5 | 74.7 | 69.06 |
| Llama-3.2-3B | AWQ (seed 1) | 72.2 | 45.6 | 73.3 | 77.0 | 69.2 | 73.4 | 68.44 |
| Llama-3.2-3B | AWQ (seed 2) | 72.0 | 45.7 | 73.2 | 76.5 | 69.7 | 74.1 | 68.54 |

### Findings (zero-shot)

- **The ranking flips compared with perplexity.** By perplexity GPTQ is better than AWQ on both models. By zero-shot accuracy AWQ is as good (1B) or clearly better (3B):
  - 1B: AWQ 58.99 vs GPTQ 58.76. The 0.23-point gap is smaller than the seed std (about 0.4), so it is a tie.
  - 3B: AWQ 68.68 vs GPTQ 67.75. The 0.93-point gap is about 3× the seed std, and every AWQ seed beats every GPTQ seed (lowest AWQ 68.44 > highest GPTQ 68.11). The gain is spread over ARC-e (+1.6), ARC-c (+1.5) and BoolQ (+2.2); only PIQA favours GPTQ (−0.7).
- **AWQ on the 3B is almost lossless:** −0.21 points vs BF16, 86% of RTN's loss recovered. (AWQ seed 0 even scores above BF16, 69.06 vs 68.89. That is noise, not an improvement.)
- **GPTQ barely beats RTN on the 3B:** +0.31 points, about one seed std, although it recovers 35% of RTN's perplexity loss. Seeds 1 and 2 are only 0.1–0.2 points above RTN.
- **Perplexity alone would have given the wrong conclusion on the 3B.** Both metrics are needed.
- **Possible explanation (hypothesis, not tested yet):** GPTQ fits each layer's output on the calibration samples (C4 web text). This is close to what perplexity measures: predicting the next word of ordinary text (WikiText-2). The zero-shot tasks are a different kind of input (short questions and answer options), and AWQ, which fits the calibration data much less, may transfer better. This matches the perplexity seed study (GPTQ depends on the calibration samples, AWQ hardly) and the AWQ paper's claim that it generalizes better. A cheap test: perplexity on C4 itself. If GPTQ's lead over AWQ is larger on C4 than on WikiText-2, GPTQ is fitting the calibration data.
- **RTN breaks BoolQ on the 1B:** 63.7 → 48.3%. That is below a coin flip, and below the 62.2% that answering "yes" to every question would get (62.2% of BoolQ answers are "yes"). BoolQ alone is about half of RTN's 5.3-point loss on the 1B; the other five tasks lose 3.3 points on average. Even BF16 1B (63.7) is only just above "always yes", so on the 1B this task mostly measures whether the model leans towards "yes" or "no", and a small change in the weights shifts that lean for many questions at once. This is also why BoolQ varies most between seeds (1B: GPTQ 58.5–62.5, AWQ 57.7–60.9).
- **Zero-shot is much noisier than perplexity.** The seed std of the average is 0.31–0.45 points for both methods. AWQ's perplexity barely changes between seeds (std 0.003–0.005), but its zero-shot average still moves by up to 0.8 points (1B: 59.31 vs 58.48), mostly through BoolQ. Tiny weight changes flip the questions where two answers are almost equally likely. Rule of thumb for this setup: differences below about 0.5 points in the average are not meaningful for a single checkpoint.
- **The larger model is more robust here too:** RTN loses 5.32 points on the 1B but 1.45 on the 3B; the best method loses 1.45 vs 0.21.
- **Batch size shifts the scores slightly, in no fixed direction:** all eight 1B models were also scored at batch 32 (before the out-of-memory stop). The averages differ from batch 16 by −0.26 to +0.16 points (mean −0.06): BF16 +0.16, RTN −0.15, GPTQ −0.01 / −0.06 / −0.20, AWQ +0.09 / −0.26 / −0.07 (seeds 0 / 1 / 2). Padding changes the shapes of the BF16 matrix multiplies, so the rounding differs and near-tie questions flip. The conclusions do not change: the 1B GPTQ–AWQ gap is 0.22 points at batch 32 and 0.23 at batch 16. All reported numbers use batch 16 for every model.

## Quantization cost

- **AWQ is the most expensive:** about 3× the time of GPTQ and 4× the GPU memory, on both models. This goes against the AWQ paper's claim of lower cost. In llm-compressor, AWQ re-runs each block for 20 candidate scales on all 262k calibration tokens and keeps the block inputs on the GPU, while GPTQ needs one pass per layer to collect statistics. Our 2048-token samples (AWQ's default is 512) make this heavier. Cost depends on the implementation and settings.
- **Cost grows roughly with model size** (1B → 3B is 2.6× more parameters): GPTQ takes 2.3× longer, AWQ 2.7× longer.
- **RTN is almost free** (2–3 s, no calibration data).

## Speed and memory (vLLM)

Full measurements (medians; see experimental_setup.md for the workloads):

| Model | Method | Weights in GPU (GiB) | KV cache (k tokens) | TTFT, 128 tokens (ms) | TPOT, 1 request (ms) | Decode tok/s, 1 request | Prefill tok/s (32 × 1024) | Throughput out tok/s (256 req.) | GPU TFLOPS before / after |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | BF16 | 2.32 | 540 | 7.97 | 4.500 | 222.2 | 43,445 | 8,755 | 106.7 / 101.9 |
| Llama-3.2-1B | RTN | 0.99 | 568 | 7.36 | 2.362 | 423.4 (1.91×) | 41,731 (0.96×) | 8,616 (0.98×) | 104.4 / 101.3 |
| Llama-3.2-1B | GPTQ | 0.99 | 570 | 7.30 | 2.353 | 425.0 (1.91×) | 41,668 (0.96×) | 8,673 (0.99×) | 106.3 / 102.8 |
| Llama-3.2-1B | AWQ | 0.99 | 568 | 7.35 | 2.351 | 425.3 (1.91×) | 41,456 (0.95×) | 8,633 (0.99×) | 106.4 / 101.4 |
| Llama-3.2-3B | BF16 | 6.04 | 113 | 16.49 | 11.188 | 89.4 | 16,703 | 2,535 | 106.5 / 102.5 |
| Llama-3.2-3B | RTN | 2.15 | 144 | 13.15 | 4.778 | 209.3 (2.34×) | 15,510 (0.93×) | 2,586 (1.02×) | 106.4 / 102.7 |
| Llama-3.2-3B | GPTQ | 2.15 | 144 | 13.17 | 4.787 | 208.9 (2.34×) | 15,531 (0.93×) | 2,592 (1.02×) | 104.1 / 104.3 |
| Llama-3.2-3B | AWQ | 2.15 | 144 | 13.16 | 4.784 | 209.0 (2.34×) | 15,529 (0.93×) | 2,592 (1.02×) | 106.6 / 103.0 |

### Decode speed vs. the memory-bandwidth limit

When one request generates tokens, every token requires reading all the weights once. So the time per token is bounded by weight bytes / memory bandwidth (A40: 696 GB/s). W4A16 rows use the mean TPOT of RTN, GPTQ and AWQ.

| Model | Format | Weights read per token | Measured TPOT | Effective bandwidth | Share of 696 GB/s | Speedup: predicted (byte ratio) → measured |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | BF16 | 2.49 GB | 4.50 ms | 554 GB/s | 80% | |
| Llama-3.2-1B | W4A16 | 1.06 GB | 2.36 ms | 451 GB/s | 65% | 2.34× → 1.91× (82% of predicted) |
| Llama-3.2-3B | BF16 | 6.49 GB | 11.19 ms | 580 GB/s | 83% | |
| Llama-3.2-3B | W4A16 | 2.31 GB | 4.78 ms | 483 GB/s | 69% | 2.81× → 2.34× (83% of predicted) |

### Findings (speed and memory)

- **Single-request generation is much faster with 4-bit weights:** 1.91× on the 1B and 2.34× on the 3B. Both reach about 82% of the speedup predicted from the byte ratio. The rest is fixed work that does not shrink with the weights (attention, normalization, the BF16 `lm_head`, kernel launches, sampling), plus Marlin's unpacking cost.
- **Why the 3B gains more:** mostly because its byte ratio is better (2.81× vs 2.34×, since the BF16 embedding is a smaller share), not because of overheads. Both sizes reach the same share of their predicted speedup.
- **Memory-bound confirmed:** BF16 generation runs at 80–83% of the GPU's memory bandwidth, so reading weights really is the bottleneck.
- **A 4-bit 3B is about as fast and as small as a BF16 1B, and much more accurate:** 209 vs 222 tok/s, 2.15 vs 2.32 GiB of weights, perplexity 8.25 vs 9.76. For a fixed speed or memory budget, a larger quantized model beats a smaller full-precision one. This is a key practical argument for quantization.
- **No gain when compute is the bottleneck:** prefill is 4–7% *slower* with W4A16 (0.93–0.96×), and throughput under heavy load is unchanged (0.98–1.02×). In both cases the GPU is busy with arithmetic: prefill runs at an estimated 80–90% of the measured BF16 matmul speed (estimate: 2 × non-embedding parameters FLOPs per token, plus attention). W4A16 still does BF16 math, plus the unpacking. **To speed up these workloads, activations must be quantized too (W8A8, W4A4), so that the INT8/INT4 tensor cores can be used.** This motivates groups B and C.
- **Short prompts still benefit a little:** TTFT for a 128-token prompt drops 8% (1B) and 20% (3B). At this size the prompt is too short to keep the GPU busy with arithmetic, so weight reading still matters.
- **More room for the KV cache:** +5% tokens on the 1B and +27% on the 3B. On the 3B, BF16 can hold about 147 full requests (512 + 256 tokens) at once, W4A16 about 187. Throughput still only rises 2%, because at that batch size each step is compute-bound: running more requests at once makes each step longer. The extra KV room would matter more with longer contexts or a smaller GPU.
- **RTN, GPTQ and AWQ run at the same speed** (within 1%): same format, same kernel. Choose between them on accuracy and quantization cost only.
- **vLLM and Hugging Face agree:** the GPTQ checkpoint generated word-for-word the same greedy text in vLLM (Marlin kernel) as in Hugging Face, so the speed and accuracy numbers describe the same model.
- **Measurement quality:** the GPU probe read 101–107 TFLOPS around every run (an idle GPU gives about 110; sharing with the other VM would give roughly half), so there was no interference. The value after each run is consistently about 3% lower, probably because the GPU is warmer after sustained load. RTN, GPTQ and AWQ, which should be identical, agree within 0.5% on TPOT, so the noise is small.

## Log

- 2026-11-09: Llama-3.2-1B BF16, PPL 9.7564 (141 windows, torch 2.14.1+cu126, transformers 5.22.2). Matches the published value of about 9.76.
- 2026-11-09: Llama-3.2-3B BF16, PPL 7.8171 (141 windows, same versions). Matches the published value of about 7.8.
- 2026-11-09: llm-compressor 0.15.0 installed, transformers 5.22.2 → 5.21.0. BF16 re-runs: Llama-3.2-1B PPL 9.7564 and Llama-3.2-3B PPL 7.8171, both unchanged.
- 2026-11-09: Llama-3.2-1B GPTQ W4A16_ASYM g128 (C4 128×2048 calibration): PPL 10.6101 (+0.85). Quantization 72.3 s, peak GPU 2.32 GiB, peak RAM 8.82 GiB, 0.96 GiB on disk. Sample generation coherent. Evaluation in HF took 11.3 s, same as BF16: HF does not use the fast 4-bit kernels, so speed is measured in vLLM.
- 2026-11-09: Llama-3.2-1B RTN W4A16_ASYM: PPL 11.7094 (+1.95). 1.9 s, peak GPU 0.06 GiB, peak RAM 3.44 GiB, 0.96 GiB.
- 2026-11-09: Llama-3.2-1B AWQ W4A16_ASYM (duo_scaling="both", same calibration samples): PPL 10.9059 (+1.15). 215.2 s, peak GPU 9.22 GiB, peak RAM 8.86 GiB, 0.96 GiB. Sample generations coherent for all three methods.
- 2026-11-09: Llama-3.2-3B W4A16_ASYM, same calibration samples: RTN PPL 8.490 (2.9 s, 0.13 GiB GPU); GPTQ 8.241 (166.9 s, 2.74 GiB GPU); AWQ 8.282 (571.8 s, 11.64 GiB GPU). All 2.098 GiB on disk.
- 2026-11-09: recipe.yaml checked. GPTQ: block_size 128, dampening_frac 0.01, actorder static. AWQ: 4 mappings (the v_proj→o_proj one is skipped under GQA), duo_scaling "both", n_grid 20, no clipping search.
- 2026-11-09: calibration-seed study (`scripts/run_seed_study.sh`), GPTQ and AWQ with seeds 1 and 2 on both models. 1B GPTQ 10.889 / 10.598, AWQ 10.913 / 10.903; 3B GPTQ 8.260 / 8.261, AWQ 8.2817 / 8.2776. Mean ± std (summarize.py): 1B GPTQ 10.699 ± 0.164, AWQ 10.907 ± 0.005; 3B GPTQ 8.254 ± 0.011, AWQ 8.280 ± 0.003.
- 2026-11-09: vLLM 0.32.0+cu129 environment working (torchcodec removed, FlashInfer sampler off). `vllm_check.py`: GPTQ loads with MarlinLinearKernel (CompressedTensorsWNA16), weights 0.99 GiB vs 2.32 GiB BF16. Greedy sample from GPTQ in vLLM is word-for-word identical to the one from Hugging Face.
- 2026-11-09: speed benchmark (`scripts/run_speed_bench.sh`) for BF16, RTN, GPTQ, AWQ on 1B and 3B. Decode speedup 1.91× (1B) and 2.34× (3B); prefill 0.93–0.96×; throughput 0.98–1.02×. GPU probe 101–107 TFLOPS around every run.
- 2026-11-09: lm-eval 0.4.13 installed in the quant venv (torch and transformers unchanged). Test run, Llama-3.2-1B BF16 zero-shot: ARC-e 61.78, ARC-c 36.60, HellaSwag 64.24, PIQA 74.59 (acc_norm); WinoGrande 60.93, BoolQ 63.55 (acc). Average 60.28. Took 244 s. (Batch size 32.)
- 2026-11-09: full zero-shot run stopped with CUDA out of memory at batch size 32 (an 8.9 GiB log-softmax over the 128k vocabulary for the longest questions). Batch size set to 16 for every model. Batch-32 results moved to `results/zeroshot_bs32/`, and everything re-run at 16.
- 2026-11-10: zero-shot for all 16 group-A models at batch size 16 (`scripts/run_zeroshot.sh`). Averages: 1B BF16 60.44, RTN 55.12, GPTQ 58.69 / 58.42 / 59.16, AWQ 59.31 / 58.48 / 59.18 (seeds 0 / 1 / 2); 3B BF16 68.89, RTN 67.44, GPTQ 68.11 / 67.62 / 67.53, AWQ 69.06 / 68.44 / 68.54. The 1B BF16 at batch 32 scored 60.28 (+0.16 at batch 16).
- 2026-11-10: batch 32 vs 16 for all eight 1B models: differences −0.26 to +0.16 points, mean −0.06, no systematic direction.
