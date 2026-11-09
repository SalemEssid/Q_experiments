# Results

Hardware and software: see [experimental_setup.md](experimental_setup.md).

**Perplexity protocol:** WikiText-2 (raw) test split, joined and tokenized once, non-overlapping 2048-token windows, PPL = exp(mean next-token NLL). This is the same protocol as the GPTQ, AWQ, QuaRot and SpinQuant papers. Script: `scripts/ppl_eval.py`.

## Comparison table

"Format" is WxAy: weights in x bits, activations in y bits. W4A16 and W8A8 methods solve different problems (memory vs. compute), so compare them with care.

| Model | Method | Format | Group size | WikiText-2 PPL ↓ | Zero-shot avg ↑ | Model size | Peak GPU mem | Throughput (tok/s) | Latency | Quantization time |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | BF16 baseline | W16A16 | – | 9.756 | pending | pending | pending | pending | pending | n/a |
| Llama-3.2-3B | BF16 baseline | W16A16 | – | 7.817 | pending | pending | pending | pending | pending | n/a |

## Log

- 2026-11-09: Llama-3.2-1B BF16, PPL 9.7564 (141 windows, torch 2.14.1+cu126, transformers 5.22.2). Matches the published value of about 9.76.
- 2026-11-09: Llama-3.2-3B BF16, PPL 7.8171 (141 windows, same versions). Matches the published value of about 7.8.
