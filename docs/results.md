# Results

Hardware and software: see [experimental_setup.md](experimental_setup.md).

**Perplexity protocol:** WikiText-2 (raw) test split, joined and tokenized once, non-overlapping 2048-token windows, PPL = exp(mean next-token NLL). This is the same protocol as the GPTQ, AWQ, QuaRot and SpinQuant papers. Script: `scripts/ppl_eval.py`.

## Comparison table

"Format" is WxAy: weights in x bits, activations in y bits. W4A16 and W8A8 methods solve different problems (memory vs. compute), so compare them with care.

| Model | Method | Format | Group size | WikiText-2 PPL ↓ | Zero-shot avg ↑ | Model size | Peak GPU mem | Throughput (tok/s) | Latency | Quantization time |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | BF16 baseline | W16A16 | – | 9.756 | pending | 2.30 GiB* | pending | pending | pending | n/a |
| Llama-3.2-1B | RTN (reference) | W4A16 asym | 128 | 11.709 | pending | 0.96 GiB | pending | pending | pending | 1.9 s, 0.06 GiB GPU, 3.4 GiB RAM |
| Llama-3.2-1B | GPTQ | W4A16 asym | 128 | 10.610 | pending | 0.96 GiB | pending | pending | pending | 72 s, 2.3 GiB GPU, 8.8 GiB RAM |
| Llama-3.2-1B | AWQ | W4A16 asym | 128 | 10.906 | pending | 0.96 GiB | pending | pending | pending | 215 s, 9.2 GiB GPU, 8.9 GiB RAM |
| Llama-3.2-3B | BF16 baseline | W16A16 | – | 7.817 | pending | 5.98 GiB* | pending | pending | pending | n/a |
| Llama-3.2-3B | RTN (reference) | W4A16 asym | 128 | 8.490 | pending | 2.10 GiB | pending | pending | pending | 2.9 s, 0.13 GiB GPU |
| Llama-3.2-3B | GPTQ | W4A16 asym | 128 | 8.241 | pending | 2.10 GiB | pending | pending | pending | 167 s, 2.7 GiB GPU |
| Llama-3.2-3B | AWQ | W4A16 asym | 128 | 8.282 | pending | 2.10 GiB | pending | pending | pending | 572 s, 11.6 GiB GPU |

\* BF16 size = parameter count × 2 bytes. Quantized sizes are the measured `.safetensors` files.

**Model size note:** both models share one matrix between the embedding and `lm_head` (128,256 × hidden size), and it stays in BF16.

| Model | Shared embedding (BF16) | Share of the 4-bit checkpoint | Compression vs BF16 |
|---|---|---|---|
| Llama-3.2-1B | 0.49 GiB | 51% | 2.40× |
| Llama-3.2-3B | 0.73 GiB | 35% | 2.85× |

## Accuracy summary (W4A16 g128)

| Model | RTN Δ PPL | GPTQ Δ PPL | AWQ Δ PPL | GPTQ recovers | AWQ recovers | GPTQ–AWQ gap |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | +1.953 (+20.0%) | +0.854 (+8.7%) | +1.150 (+11.8%) | 56% | 41% | 0.296 |
| Llama-3.2-3B | +0.673 (+8.6%) | +0.424 (+5.4%) | +0.465 (+6.0%) | 37% | 31% | 0.041 |

"Recovers" = share of RTN's perplexity loss that the method wins back. This table uses seed 0 only; see the seed study below for averages.

## Calibration-seed study (W4A16 g128)

Same method and settings, three different random sets of 128 × 2048 C4 calibration samples (seeds 0, 1, 2). RTN uses no data, so it has no seed. Std is the sample standard deviation over the three seeds.

| Model | Method | Seed 0 | Seed 1 | Seed 2 | Mean ± std | Spread (max − min) | RTN loss recovered (mean) |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | GPTQ | 10.610 | 10.889 | 10.598 | **10.699 ± 0.165** | 0.291 | 52% |
| Llama-3.2-1B | AWQ | 10.906 | 10.913 | 10.903 | **10.907 ± 0.005** | 0.010 | 41% |
| Llama-3.2-3B | GPTQ | 8.241 | 8.260 | 8.261 | **8.254 ± 0.011** | 0.020 | 35% |
| Llama-3.2-3B | AWQ | 8.282 | 8.282 | 8.278 | **8.281 ± 0.002** | 0.004 | 31% |

Quantization time and memory are the same for every seed (within 0.5 s).

**Findings:**
- **GPTQ has the lower perplexity in every seed on both models,** but on the 3B the mean gap is only 0.027 PPL (0.3%), a tie in practice. On the 1B, GPTQ's worst seed (10.889) is only just below AWQ's best (10.903).
- **AWQ is much more stable across calibration sets:** spread 0.010 vs 0.291 on the 1B (about 30× smaller), and 0.004 vs 0.020 on the 3B (5× smaller). This supports the AWQ paper's claim that it depends less on the calibration data. GPTQ fits the layer outputs on the exact calibration samples, so an unlucky set (seed 1 on the 1B) costs it about a third of its improvement over RTN. AWQ only uses average activation sizes, which barely change between sample sets.
- **A single seed is not enough:** seed 0 was a lucky seed for GPTQ on the 1B. Results in the report should be given as mean ± std over 3 seeds.

## Observations so far (W4A16 g128, perplexity only)

- **Same ranking on both models: GPTQ < AWQ < RTN** (lower PPL is better). A likely reason: GPTQ corrects rounding errors (it updates the remaining weights after each column), while AWQ only rescales important channels before plain rounding. Also, llm-compressor's AWQ has **no weight-clipping search**, which the AWQ paper includes (see methods_notes.md).
- **The GPTQ–AWQ gap almost disappears on the 3B.** Over 3 calibration seeds: 0.21 PPL on the 1B, 0.027 on the 3B. GPTQ is ahead in every seed, but on the 3B the difference is negligible.
- **AWQ is 5–30× more stable than GPTQ across calibration sets** (see the seed study).
- **The larger model is more robust:** RTN's loss drops from +20.0% (1B) to +8.6% (3B), GPTQ's from +8.7% to +5.4%.
- **The recovered share is lower on the 3B** (37% / 31% vs 56% / 41%). When RTN already does well, much of the remaining error is the inherent cost of 4 bits, which no rounding method removes.
- **Compression is better on the 3B** (2.85× vs 2.40×), as predicted: the BF16 embedding is a smaller share of the model.
- **Quantization cost: AWQ is the most expensive** (about 3× the time of GPTQ and 4× the GPU memory, on both models). This goes against the AWQ paper's claim of lower cost. In llm-compressor, AWQ re-runs each block for 20 candidate scales on all 262k calibration tokens, while GPTQ needs one pass per layer to collect statistics. Our 2048-token samples (AWQ's default is 512) make this heavier. Cost depends on the implementation and settings.
- **Cost grows roughly with model size** (1B → 3B is 2.6× more parameters): GPTQ 2.3× longer, AWQ 2.7× longer.
- **Open question:** does the ranking hold on zero-shot tasks?

## Log

- 2026-11-09: Llama-3.2-1B BF16, PPL 9.7564 (141 windows, torch 2.14.1+cu126, transformers 5.22.2). Matches the published value of about 9.76.
- 2026-11-09: Llama-3.2-3B BF16, PPL 7.8171 (141 windows, same versions). Matches the published value of about 7.8.
- 2026-11-09: llm-compressor 0.15.0 installed, transformers 5.22.2 → 5.21.0. BF16 re-runs: Llama-3.2-1B PPL 9.7564 and Llama-3.2-3B PPL 7.8171, both unchanged.
- 2026-11-09: Llama-3.2-1B GPTQ W4A16_ASYM g128 (C4 128×2048 calibration): PPL 10.6101 (+0.85). Quantization 72.3 s, peak GPU 2.32 GiB, peak RAM 8.82 GiB, 0.96 GiB on disk. Sample generation coherent. Evaluation in HF took 11.3 s, same as BF16: HF does not use the fast 4-bit kernels, so speed is measured in vLLM.
- 2026-11-09: Llama-3.2-1B RTN W4A16_ASYM: PPL 11.7094 (+1.95). 1.9 s, peak GPU 0.06 GiB, peak RAM 3.44 GiB, 0.96 GiB.
- 2026-11-09: Llama-3.2-1B AWQ W4A16_ASYM (duo_scaling="both", same calibration samples): PPL 10.9059 (+1.15). 215.2 s, peak GPU 9.22 GiB, peak RAM 8.86 GiB, 0.96 GiB. Sample generations coherent for all three methods.
- 2026-11-09: Llama-3.2-3B W4A16_ASYM, same calibration samples: RTN PPL 8.490 (2.9 s, 0.13 GiB GPU); GPTQ 8.241 (166.9 s, 2.74 GiB GPU); AWQ 8.282 (571.8 s, 11.64 GiB GPU). All 2.098 GiB on disk.
- 2026-11-09: recipe.yaml checked. GPTQ: block_size 128, dampening_frac 0.01, actorder static. AWQ: 4 mappings (the v_proj→o_proj one is skipped under GQA), duo_scaling "both", n_grid 20, no clipping search.
- 2026-11-09: calibration-seed study (`scripts/run_seed_study.sh`), GPTQ and AWQ with seeds 1 and 2 on both models. 1B GPTQ 10.889 / 10.598, AWQ 10.913 / 10.903; 3B GPTQ 8.260 / 8.261, AWQ 8.282 / 8.278. Mean ± std: 1B GPTQ 10.699 ± 0.165, AWQ 10.907 ± 0.005; 3B GPTQ 8.254 ± 0.011, AWQ 8.281 ± 0.002.
