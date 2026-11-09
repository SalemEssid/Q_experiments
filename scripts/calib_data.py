"""Calibration data shared by every quantization method.

GPTQ, AWQ and SmoothQuant all look at a small sample of text before quantizing:
GPTQ uses it to measure which weights matter most, AWQ and SmoothQuant to find the large activations.
For a fair comparison, every method gets exactly the same samples:
  - 128 windows of 2048 tokens (the GPTQ paper's setting)
  - cut at random positions from long documents in the first shard of the C4 training set
  - fixed random seed; the result is saved to disk the first time and reloaded afterwards
C4 is general web text, not WikiText-2, so we never calibrate on the data we evaluate on.
"""
import random
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

C4_SHARD = "en/c4-train.00000-of-01024.json.gz"  # about 300 MB, downloaded once


def get_calibration_data(tokenizer, model_name, n_samples=128, seqlen=2048, seed=0, cache_dir="data/calib"):
    cache = Path(cache_dir) / f"c4_{model_name}_{n_samples}x{seqlen}_seed{seed}"
    if cache.exists():
        print(f"Calibration data: reusing {cache}")
        return load_from_disk(str(cache))

    docs = load_dataset("allenai/c4", data_files={"train": C4_SHARD}, split="train")
    rng = random.Random(seed)
    windows = []
    while len(windows) < n_samples:
        text = docs[rng.randrange(len(docs))]["text"]
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) <= seqlen:  # too short for a full window: pick another document, as in GPTQ
            continue
        start = rng.randrange(len(ids) - seqlen)
        windows.append(ids[start : start + seqlen])

    ds = Dataset.from_dict({"input_ids": windows, "attention_mask": [[1] * seqlen for _ in windows]})
    ds.save_to_disk(str(cache))
    print(f"Calibration data: built {n_samples} x {seqlen} tokens, saved to {cache}")
    return ds
