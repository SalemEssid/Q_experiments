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

## Group B: W8A8 INT8 (accuracy done; speed in progress)

`W8A8` = INT8 symmetric weights (one scale per output channel) and INT8 symmetric activations (one scale per token, computed at run time). `lm_head` stays BF16. Plan and settings: experiment_plan.md and methods_notes.md, section 6.

| Model | Method | Scheme | WikiText-2 PPL ↓ | Δ vs BF16 | Zero-shot avg ↑ | Δ vs BF16 (points) | Size on disk | Quantization cost |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | BF16 | – | 9.756 | – | 60.44 | – | 2.30 GiB | – |
| Llama-3.2-1B | RTN | W8A8 | **9.888** | +0.132 (+1.4%) | 59.95 | −0.49 | 1.40 GiB | 1.6 s, 0.03 GiB GPU |
| Llama-3.2-1B | SmoothQuant α=0.5 | W8A8 | 9.903 ± 0.003 | +0.147 (+1.5%) | **60.48 ± 0.06** | +0.04 | 1.40 GiB | 28 s, 2.12 GiB GPU |
| Llama-3.2-1B | SmoothQuant α=0.8 | W8A8 | 9.890 | +0.134 (+1.4%) | 60.24 | −0.20 | 1.40 GiB | 28 s, 2.12 GiB GPU |
| Llama-3.2-1B | RTN | W8A8_STATIC | 1301.9 | broken | – | – | 1.40 GiB | 28.6 s, 2.12 GiB GPU |
| Llama-3.2-1B | SmoothQuant α=0.5 | W8A8_STATIC | 1341.8 | broken | – | – | 1.40 GiB | 53.2 s, 2.12 GiB GPU |
| Llama-3.2-1B | SmoothQuant α=0.8 | W8A8_STATIC | 1322.2 | broken | – | – | 1.40 GiB | 52.8 s, 2.12 GiB GPU |
| Llama-3.2-3B | BF16 | – | 7.817 | – | 68.89 | – | 5.98 GiB | – |
| Llama-3.2-3B | RTN | W8A8 | 7.889 | +0.072 (+0.9%) | 68.38 | −0.51 | 3.36 GiB | 2.4 s, 0.05 GiB GPU |
| Llama-3.2-3B | SmoothQuant α=0.5 | W8A8 | 7.890 ± 0.001 | +0.073 (+0.9%) | 68.50 ± 0.13 | −0.39 | 3.36 GiB | 65 s, 2.46 GiB GPU |
| Llama-3.2-3B | SmoothQuant α=0.8 | W8A8 | **7.881** | +0.064 (+0.8%) | **68.78** | −0.11 | 3.36 GiB | 65 s, 2.46 GiB GPU |
| Llama-3.2-3B | RTN | W8A8_STATIC | 60.6 | broken | – | – | 3.36 GiB | 66.0 s, 2.46 GiB GPU |

- SmoothQuant α=0.5: mean ± sample std over calibration seeds 0, 1, 2. α=0.8 and the static models: seed 0 only. RTN `W8A8` uses no calibration data. Bold: best quantized result per model and metric (all differences between the `W8A8` rows are small, see the findings).
- `W8A8_STATIC` = the same INT8 weights, but one fixed activation scale per layer measured on the calibration data (llm-compressor's `minmax` observer). Static RTN needs a calibration pass too, hence 28.6 s / 66.0 s. Peak CPU RAM on the 1B: 3.5 GiB (RTN), 8.4–8.5 GiB (SmoothQuant, which holds the calibration activations).
- Compression vs BF16: 1.65× (1B) and 1.78× (3B). The INT8 checkpoint holds the 8-bit weights plus the BF16 embedding (0.49 / 0.73 GiB).

**Activation quantization is really applied in evaluation:** `ppl_eval.py` found 112 layers with INT8 input activations on the 1B (16 blocks × 7 Linear layers; `lm_head` excluded). The first-window loss is 1.99390 with activation quantization and 1.98569 without it. The check runs on every evaluation.

### Error split: weights vs activations

"Weights only" = the same checkpoint evaluated with activation quantization switched off (`ppl_eval.py --weights-only`), i.e. W8A16.

| Model | Method | BF16 PPL | Weights only PPL | W8A8 PPL | Loss from weights | Extra loss from activations |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | RTN | 9.756 | 9.779 | 9.888 | +0.023 | +0.109 (83%) |
| Llama-3.2-1B | SmoothQuant α=0.5 (seeds 0 / 1 / 2) | 9.756 | 9.778 / 9.780 / 9.780 | 9.904 / 9.899 / 9.905 | +0.022 / +0.023 / +0.023 | +0.126 / +0.120 / +0.126 |
| Llama-3.2-1B | SmoothQuant α=0.8 | 9.756 | 9.782 | 9.890 | +0.026 | +0.108 |
| Llama-3.2-3B | RTN | 7.817 | 7.823 | 7.889 | +0.006 | +0.066 (92%) |
| Llama-3.2-3B | SmoothQuant α=0.5 (seeds 0 / 1 / 2) | 7.817 | 7.825 / 7.824 / 7.822 | 7.891 / 7.889 / 7.891 | +0.008 / +0.007 / +0.005 | +0.066 / +0.065 / +0.069 |
| Llama-3.2-3B | SmoothQuant α=0.8 | 7.817 | 7.825 | 7.881 | +0.008 | **+0.056** |

### Zero-shot per task

Same tasks and settings as group A (batch size 16). SmoothQuant α=0.5: mean of the three seeds (from the 1-decimal values, so ±0.05).

| Model | Method | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | BoolQ | Avg |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | BF16 | 61.8 | 36.9 | 64.3 | 74.7 | 61.3 | 63.7 | 60.44 |
| Llama-3.2-1B | RTN W8A8 | 62.1 | 36.3 | 63.8 | 75.0 | 59.8 | 62.8 | 59.95 |
| Llama-3.2-1B | SmoothQuant α=0.5 | 62.2 | 37.2 | 63.9 | 74.6 | 61.3 | 63.6 | 60.48 ± 0.06 |
| Llama-3.2-1B | SmoothQuant α=0.8 | 62.1 | 36.7 | 63.9 | 74.1 | 61.0 | 63.6 | 60.24 |
| Llama-3.2-3B | BF16 | 71.9 | 46.1 | 74.1 | 77.8 | 69.8 | 73.7 | 68.89 |
| Llama-3.2-3B | RTN W8A8 | 71.5 | 46.1 | 73.9 | 77.2 | 69.7 | 72.0 | 68.38 |
| Llama-3.2-3B | SmoothQuant α=0.5 | 71.3 | 46.5 | 73.8 | 77.3 | 69.2 | 72.9 | 68.50 ± 0.13 |
| Llama-3.2-3B | SmoothQuant α=0.8 | 71.6 | 46.2 | 73.9 | 77.5 | 69.8 | 73.7 | 68.78 |

SmoothQuant α=0.5 per seed: 1B 60.52 / 60.41 / 60.50, 3B 68.65 / 68.41 / 68.45.

### Findings (accuracy, `W8A8` with per-token scales)

- **8-bit is close to lossless on both models, even with plain rounding:** RTN loses +0.132 PPL (1B) and +0.072 (3B), about 7× and 6× less than the best W4A16 method (GPTQ: +0.94 / +0.44). On zero-shot, RTN loses about 0.5 points on both models, which is the size of the noise. BoolQ on the 1B stays at 62.8, while RTN W4A16 broke it (48.3).
- **Most of the loss comes from the activations:** 83% (1B) and 92% (3B). 8-bit weights alone cost only +0.023 / +0.006 PPL.
- **SmoothQuant brings no reliable gain:**
  - α=0.5 (the main setting): on perplexity, slightly worse than RTN on the 1B (9.903 vs 9.888; all three seeds are worse) and equal on the 3B (7.890 vs 7.889). On zero-shot, +0.53 points on the 1B and +0.12 on the 3B. The 1B gain is mostly WinoGrande (+1.5, about one standard error of that task) and runs against the perplexity, so it is not a clear improvement.
  - α=0.8 (sensitivity check): equal to RTN on the 1B (9.890). Slightly better on the 3B: 7.881, which wins back 11% of RTN's loss, and the activation loss drops from +0.066 to +0.056. This is the only case where smoothing helped with per-token scales. It is a single seed, but the gap (0.008) is 8× the seed std of α=0.5. Its zero-shot average (68.78) is within noise of the others. α=0.5 stays the main setting, as decided in advance (choosing α on these test results would tune on the test set).
  - **Reason:** the largest part of the activation loss sits in `down_proj`'s input (70% on the 1B, layer test below), which SmoothQuant does not smooth on Llama. Per-token scales already handle the extreme tokens.
- **SmoothQuant hardly depends on the calibration samples:** PPL std 0.003 (1B) and 0.001 (3B), zero-shot std 0.06 and 0.13. This matches AWQ (std 0.005 / 0.003). Both methods only use per-channel activation sizes, which barely change between sample sets.
- **The 3B is more robust again:** +0.9% PPL vs +1.4% on the 1B.
- **Quantization cost:** SmoothQuant takes 28 s (1B) and 65 s (3B), with 2.1–2.5 GiB of GPU memory: one calibration pass to collect channel ranges. That is cheaper than GPTQ (72 / 167 s) and AWQ (215 / 572 s) at W4A16. RTN takes 2 s.
- **Compared with W4A16 (accuracy only; the two formats save different things):** on the 1B, even RTN W8A8 (+0.13 PPL, −0.49 points) beats every W4A16 method (best: +0.94 PPL, −1.45 points). On the 3B, perplexity separates them clearly (+0.07 vs +0.44), but on zero-shot AWQ W4A16 (−0.21) and W8A8 (−0.11 to −0.51) are all within noise of BF16. Zero-shot cannot separate losses this small; perplexity can.

### Which layers cause the activation loss (`W8A8`, per-token scales)

`scripts/run_act_diag.sh`: one group of layers gets BF16 inputs, all others stay W8A8. "Share" = part of the activation loss (W8A8 PPL − weights-only PPL) that disappears.

| Model | Method | BF16 inputs in | PPL | PPL drop | Share of activation loss |
|---|---|---|---|---|---|
| Llama-3.2-1B | RTN (9.888, activation loss +0.109) | q/k/v (smoothed by SQ) | 9.875 | 0.013 | 12% |
| | | o_proj | 9.884 | 0.004 | 4% |
| | | gate/up (smoothed by SQ) | 9.879 | 0.009 | 9% |
| | | **down_proj** | **9.812** | **0.076** | **70%** |
| Llama-3.2-1B | SmoothQuant α=0.5 (9.904, activation loss +0.126) | q/k/v (smoothed) | 9.882 | 0.022 | 18% |
| | | o_proj | 9.898 | 0.006 | 5% |
| | | gate/up (smoothed) | 9.882 | 0.022 | 18% |
| | | **down_proj** | **9.827** | **0.077** | **62%** |

- **`down_proj`'s input causes 70% of the activation loss**, and SmoothQuant does not touch it: its cost is the same with and without smoothing (0.076 vs 0.077). The four shares add up to about 100% (95% and 103%), so the groups contribute roughly independently.
- **In the layers SmoothQuant does smooth, it made the activation loss larger:** q/k/v 0.013 → 0.022, gate/up 0.009 → 0.022. With one scale per token, what matters is how uneven the channels are *within each token*. SmoothQuant's scales are chosen from each channel's extreme over all calibration tokens, which apparently does not make an ordinary token more even (hypothesis; `scripts/act_stats.py` measures the activations).

### Static activation scales break the model

- **All three static models are broken:** PPL 1302 (RTN), 1342 (SQ α=0.5), 1322 (SQ α=0.8), against 9.89 with per-token scales. Same weights, so this is entirely the activation scale. The sample generations are degraded too: wrong facts ("located in the north-east corner of the country") and repetition ("a a a a" for SQ α=0.5).
- **SmoothQuant does not rescue it,** although this is the setting it was designed for.

**Cause, confirmed: massive activations in the input of `down_proj`.**

Layer test on the static models: with BF16 inputs in `down_proj` only, perplexity comes back from 1302 to **9.987** (RTN) and from 1342 to **9.929** (SmoothQuant). Leaving q/k/v, o_proj or gate/up in BF16 changes nothing (1300–1373). All of the damage is in `down_proj`.

Activation measurements (`scripts/act_stats.py`, BF16 Llama-3.2-1B, WikiText-2 test window 1 = tokens 2048–4095). "Typical" = the median over tokens of each token's largest |value|:

| Block | Max abs value in `down_proj` input | At token (position) | Typical token max | Max / typical | Without that token |
|---|---|---|---|---|---|
| 0 | 41.2 | ' ' (85) | 0.43 | 95 | 11 |
| **1** | **1240.0** | ' service' (**0**) | 0.47 | **2613** | 65 |
| 2–13 | 2.4–9.6 | various | 0.65–2.62 | 4–6 | 3–5 |
| 14 | 12.2 | ' service' (0) | 3.45 | 4 | 3 |
| **15** | **378.0** | ' service' (**0**) | 4.59 | **82** | 7 |

- **One token, not one channel:** in block 1 the most extreme value is 2613× an ordinary token's range, but only 65× once that single token is left out. These are the "massive activations" described by Sun et al. (*Massive Activations in Large Language Models*, 2024; check the exact claims before citing).
- **The token is just the first position of the window,** an ordinary word (" service"), with no BOS token. The huge value is tied to position 0, not to a special token. The calibration windows also start at an arbitrary token, so the calibrated static scales see the same effect.
- **Why one static scale fails:** with the scale set by the extreme (scale ≈ max/127), every value below max/254 rounds to zero. In block 1 an ordinary token's largest value is 1/2613 of the max, so **all of its `down_proj` inputs become 0**. Blocks 0 and 15 (ratio ≈ 90) leave ordinary tokens about 1–2 levels. Per-token scales do not have this problem: the extreme token gets its own scale.
- **SmoothQuant leaves this input unchanged:** the same measurement on the SmoothQuant α=0.5 checkpoint (quantization switched off) gives the same numbers (block 1: 1240.0, ratio 2602; the small differences come from its INT8 weights). Smoothing is an exact transformation, and `down_proj` is not among the inputs it smooths.
- **The calibrated static scales match the measured extremes:** the scale saved in the checkpoint for each `down_proj` input covers a largest value of 127 × scale. That is 41.7 (block 0), **1191 (block 1)**, 2.5–13 (blocks 2–14) and 363 (block 15), against 41.2, 1240, 2.4–12 and 378 measured on WikiText-2. Calibration (C4) found the same massive values, so they are a property of the model, not of the data. In block 1, every value below 4.69 rounds to 0, while an ordinary token's *largest* value is about 0.47: that layer receives only zeros for every ordinary token.
- **Three layers out of 112 cause the collapse.** Static RTN with BF16 inputs in only some `down_proj` layers:

  | `down_proj` inputs kept in BF16 | Layers | PPL |
  |---|---|---|
  | none (all static) | 0 of 112 | 1301.9 |
  | block 1 | 1 | 71.97 |
  | blocks 0, 1 and 15 | 3 | **10.891** |
  | all 16 blocks | 16 | 9.987 |
  | *reference: per-token scales in every layer* | – | *9.888* |

- **For `down_proj`, per-token scales matter even without the massive tokens:** the 13 other `down_proj` layers, with no extreme values (ratio 4–6), still cost +0.90 PPL with static scales (10.891 vs 9.987). Their input is spiky within each token, so a per-token scale is several times finer than one fixed scale (block 2: a typical token's own scale would be about 0.72/127 ≈ 0.0057, the static one is 0.0229).
- **But SmoothQuant does help the layers it reaches, in the static setting:** with `down_proj` excluded, static scales on the other layers cost RTN +0.175 PPL (9.987 vs 9.812 with per-token scales and BF16 `down_proj` inputs), but SmoothQuant only +0.102 (9.929 vs 9.827). That is 42% less, which is what the method is for. With per-token scales there was nothing for it to fix.
- Hugging Face evaluation takes 18 s instead of 11 s, because the activation rounding is simulated in software. This says nothing about real speed (measured in vLLM).

**The 3B: same failure, same cause.**

- Static RTN on the 3B: PPL **60.6**, against 7.889 with per-token scales. Less extreme than the 1B (1302), but still unusable.
- With BF16 inputs in `down_proj` only, it comes back to **7.948**. So, as on the 1B, practically all of the damage is in `down_proj` (99.8% of the activation loss, taking the weights-only PPL 7.823 of RTN `W8A8`, which has the same INT8 weights; on the 1B the two checkpoints gave the same weights-only PPL, 9.779).
- Activation measurements, BF16 Llama-3.2-3B, same window as the 1B:

  | Block | Max abs value in `down_proj` input | At token (position) | Typical token max | Max / typical | Without that token |
  |---|---|---|---|---|---|
  | 0 | 3.5 | ' the' (34) | 0.46 | 8 | 6 |
  | **1** | **732.0** | ' service' (**0**) | 0.52 | **1420** | 10 |
  | 2–26 | 3.0–21.5 | various (position 0 in blocks 12, 13 and 25) | 0.71–3.88 | 4–8 | 4–6 |
  | **27** (last) | **536.0** | ' service' (**0**) | 8.88 | **60** | 6 |

- **The same pattern as the 1B:** massive activations at position 0 in the second block (block 1) and in the last block. Unlike the 1B, block 0 has none (3.5, ratio 8 vs 41.2, ratio 95). So the 3B has two extreme `down_proj` inputs out of 28, the 1B three out of 16.
- In block 1 the ratio (1420) is far above 254, so with a scale set by the extreme, every ordinary token's `down_proj` input rounds to zero, as on the 1B. (This assumes the calibrated 3B scales again match the measured maxima; they were only checked on the 1B.)
- **Why the 3B is less broken (hypothesis, not tested):** it has one extreme layer fewer, the extremes of its last block are milder (ratio 60 vs 82), and with 28 blocks a single zeroed MLP output is a smaller part of the whole computation. The 3B's ordinary tokens are also more even: without the extreme token, every block's ratio is 4–10, against up to 65 on the 1B.

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
- 2026-11-10: group B started. Llama-3.2-1B RTN W8A8: PPL 9.8883 (+0.132). Quantization 1.6 s, peak GPU 0.03 GiB, peak RAM 3.48 GiB, 1.396 GiB on disk. Sample generation coherent. Activation check: 112 layers; first-window loss 1.99390 (activations quantized) vs 1.98569 (not quantized).
- 2026-11-10: Llama-3.2-1B SmoothQuant W8A8 (seed 0): α=0.5 PPL 9.9044 (28.1 s, 2.12 GiB GPU, 8.43 GiB RAM); α=0.8 PPL 9.8903 (28.0 s, 2.12 GiB GPU, 8.53 GiB RAM). Samples coherent. Weights-only PPL: RTN 9.7793, SQ α=0.5 9.7784, SQ α=0.8 9.7822.
- 2026-11-10: layer test (`run_act_diag.sh`) on RTN and SQ α=0.5 W8A8: BF16 inputs in down_proj remove 70% / 62% of the activation loss; q/k/v 12% / 18%, gate/up 9% / 18%, o_proj 4% / 5%.
- 2026-11-10: Llama-3.2-1B W8A8_STATIC (seed 0): RTN PPL 1301.9 (28.6 s), SQ α=0.5 1341.8 (53.2 s), SQ α=0.8 1322.2 (52.8 s); 2.12 GiB GPU each. All broken; samples degraded.
- 2026-11-10: static layer test: BF16 inputs in down_proj → RTN 9.987, SQ 9.929; other groups 1300–1373. Weights-only PPL of the static checkpoints: 9.779 (RTN), 9.778 (SQ), same as the dynamic ones. `act_stats.py` (BF16 and SQ α=0.5, window 1): down_proj input of block 1 has max 1240 at position 0, 2613× (2602× for SQ) the typical token range, 65× without that token; blocks 0 and 15 about 90×.
- 2026-11-10: static RTN with BF16 inputs in block 1's down_proj only: PPL 71.97; in blocks 0, 1, 15: 10.891. Saved down_proj input scales: block 0 0.328, block 1 9.375, block 15 2.859, blocks 2–14 0.020–0.105.
- 2026-11-10: `run_groupB.sh` finished. 1B SQ α=0.5 seeds 1 and 2: PPL 9.899 / 9.905 (27.9 / 28.0 s); weights-only 9.780 / 9.780. 3B RTN W8A8: PPL 7.889 (2.4 s, 0.05 GiB GPU), weights-only 7.823. 3B SQ α=0.5 seeds 0 / 1 / 2: 7.891 / 7.889 / 7.891 (65 s, 2.46 GiB GPU), weights-only 7.825 / 7.824 / 7.822. 3B SQ α=0.8: 7.881, weights-only 7.825. All 3B W8A8 checkpoints 3.361 GiB.
- 2026-11-10: 3B RTN W8A8_STATIC: PPL 60.612 (66.0 s, 2.46 GiB GPU). With BF16 down_proj inputs: 7.948. `act_stats.py` on BF16 3B: down_proj input max 732 in block 1 (position 0, ratio 1420) and 536 in block 27 (position 0, ratio 60); other blocks ratio 4–8.
- 2026-11-10: zero-shot of all W8A8 models (batch 16). 1B: RTN 59.95, SQ α=0.5 60.52 / 60.41 / 60.50, α=0.8 60.24. 3B: RTN 68.38, SQ α=0.5 68.65 / 68.41 / 68.45, α=0.8 68.78.
