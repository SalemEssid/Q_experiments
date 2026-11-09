"""Check that vLLM runs on this server (driver 535) and can load our quantized checkpoints.

Usage (from ~/Q_experiments, with the vllm venv active):
  python scripts/vllm_check.py --model meta-llama/Llama-3.2-1B
  python scripts/vllm_check.py --model models/Llama-3.2-1B-GPTQ-W4A16_ASYM
"""
import argparse
import os

# FlashInfer would compile its sampling kernels with the system nvcc 12.0, which is too old
# (it rejects --compress-mode=size). Use vLLM's own PyTorch sampler instead. Must be set before importing vLLM.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
import vllm
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    print("vLLM   :", vllm.__version__)
    print("PyTorch:", torch.__version__, "| CUDA (torch):", torch.version.cuda)

    # max_model_len=2048 keeps the KV cache small; 0.85 leaves some GPU memory free.
    llm = LLM(model=args.model, max_model_len=2048, gpu_memory_utilization=0.85, seed=0)
    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=20, temperature=0))
    print("\nSample generation:", out[0].outputs[0].text)


# vLLM starts helper processes; this guard stops them from re-running the script.
if __name__ == "__main__":
    main()
