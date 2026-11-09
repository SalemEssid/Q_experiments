# Results

Hardware and software: see [experimental_setup.md](experimental_setup.md).

**Perplexity protocol:** WikiText-2 (raw) test split, joined and tokenized once, non-overlapping 2048-token windows, PPL = exp(mean next-token NLL). This is the same protocol as the GPTQ, AWQ, QuaRot and SpinQuant papers. Script: `scripts/ppl_eval.py`.

## Comparison table

"Format" is WxAy: weights in x bits, activations in y bits. W4A16 and W8A8 methods solve different problems (memory vs. compute), so compare them with care.

| Model | Method | Format | Group size | WikiText-2 PPL ↓ | Zero-shot avg ↑ | Model size | Peak GPU mem | Throughput (tok/s) | Latency | Quantization time |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | BF16 baseline | W16A16 | – | 9.756 | pending | 2.30 GiB* | pending | pending | pending | n/a |
| Llama-3.2-1B | GPTQ | W4A16 asym | 128 | 10.610 | pending | 0.96 GiB | pending | pending | pending | 72 s, 2.3 GiB GPU, 8.8 GiB RAM |
| Llama-3.2-3B | BF16 baseline | W16A16 | – | 7.817 | pending | 5.98 GiB* | pending | pending | pending | n/a |

\* BF16 size = parameter count × 2 bytes. Quantized sizes are the measured `.safetensors` files.

Model size note: Llama-3.2-1B shares one 128,256 × 2048 matrix between the embedding and `lm_head`. It stays in BF16 (about 0.49 GiB), so it is about half of the 4-bit checkpoint. This is why compression is 2.4×, not 4×.

## Log

- 2026-11-09: Llama-3.2-1B BF16, PPL 9.7564 (141 windows, torch 2.14.1+cu126, transformers 5.22.2). Matches the published value of about 9.76.
- 2026-11-09: Llama-3.2-3B BF16, PPL 7.8171 (141 windows, same versions). Matches the published value of about 7.8.
- 2026-11-09: llm-compressor 0.15.0 installed, transformers 5.22.2 → 5.21.0. Llama-3.2-1B BF16 re-run: PPL 9.7564, unchanged.
- 2026-11-09: Llama-3.2-1B GPTQ W4A16_ASYM g128 (C4 128×2048 calibration): PPL 10.6101 (+0.85). Quantization 72.3 s, peak GPU 2.32 GiB, peak RAM 8.82 GiB, 0.96 GiB on disk. Sample generation coherent. Evaluation in HF took 11.3 s, same as BF16: HF does not use the fast 4-bit kernels, so speed is measured in vLLM.
