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

\* BF16 size = parameter count × 2 bytes. Quantized sizes are the measured `.safetensors` files.

Model size note: Llama-3.2-1B shares one 128,256 × 2048 matrix between the embedding and `lm_head`. It stays in BF16 (about 0.49 GiB), so it is about half of the 4-bit checkpoint. This is why compression is 2.4×, not 4×.

## Observations so far (Llama-3.2-1B, W4A16 g128)

- **Accuracy ranking: GPTQ < AWQ < RTN** (lower PPL is better). RTN loses 1.95 PPL. GPTQ recovers 56% of that loss, AWQ 41%. A likely reason: GPTQ corrects rounding errors (it updates the remaining weights after each column), while AWQ only rescales important channels before plain rounding.
- **Quantization cost: AWQ is the most expensive here** (215 s and 9.2 GiB GPU, vs. 72 s and 2.3 GiB for GPTQ). This goes against the AWQ paper's claim of lower cost. Likely causes are llm-compressor's scale grid search with `duo_scaling="both"` and our 2048-token calibration samples (AWQ's default is 512). Cost depends on the implementation and settings.
- **All three checkpoints are 0.96 GiB:** same format, only the rounding decisions differ, so this is an accuracy comparison at equal memory.
- **Open questions:** does the ranking hold on the 3B model and on downstream tasks? Is the 0.3 PPL gap between GPTQ and AWQ larger than the variation from changing the calibration seed?

## Log

- 2026-11-09: Llama-3.2-1B BF16, PPL 9.7564 (141 windows, torch 2.14.1+cu126, transformers 5.22.2). Matches the published value of about 9.76.
- 2026-11-09: Llama-3.2-3B BF16, PPL 7.8171 (141 windows, same versions). Matches the published value of about 7.8.
- 2026-11-09: llm-compressor 0.15.0 installed, transformers 5.22.2 → 5.21.0. BF16 re-runs: Llama-3.2-1B PPL 9.7564 and Llama-3.2-3B PPL 7.8171, both unchanged.
- 2026-11-09: Llama-3.2-1B GPTQ W4A16_ASYM g128 (C4 128×2048 calibration): PPL 10.6101 (+0.85). Quantization 72.3 s, peak GPU 2.32 GiB, peak RAM 8.82 GiB, 0.96 GiB on disk. Sample generation coherent. Evaluation in HF took 11.3 s, same as BF16: HF does not use the fast 4-bit kernels, so speed is measured in vLLM.
- 2026-11-09: Llama-3.2-1B RTN W4A16_ASYM: PPL 11.7094 (+1.95). 1.9 s, peak GPU 0.06 GiB, peak RAM 3.44 GiB, 0.96 GiB.
- 2026-11-09: Llama-3.2-1B AWQ W4A16_ASYM (duo_scaling="both", same calibration samples): PPL 10.9059 (+1.15). 215.2 s, peak GPU 9.22 GiB, peak RAM 8.86 GiB, 0.96 GiB. Sample generations coherent for all three methods.
