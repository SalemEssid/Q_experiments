"""Export our C4 calibration windows as plain token-ID tensors for the research codebases (group C).

The research codebases cannot read the datasets folders that calib_data.py saves (they pin older `datasets`
versions), so each set is also written as one LongTensor [n_samples, 2048] with torch.save:
  data/calib/c4_<model>_128x2048_seed0.pt   GPTQ calibration: the same 128 windows as groups A and B
  data/calib/c4_<model>_800x2048_seed0.pt   SpinQuant rotation training (its scripts use 800 windows)
The 800 windows are drawn by the same random sequence, so their first 128 are exactly the GPTQ windows (checked).

Usage (from ~/Q_experiments, quant venv; a few minutes for the 800 windows):
  python scripts/export_calib.py
"""
from pathlib import Path

import torch
from transformers import AutoTokenizer

from calib_data import get_calibration_data

SEQLEN = 2048


def main():
    for model in ["Llama-3.2-1B", "Llama-3.2-3B"]:
        tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{model}")
        sets = {}
        for n in [128, 800]:
            ds = get_calibration_data(tokenizer, model, n_samples=n, seqlen=SEQLEN, seed=0)
            ids = torch.tensor(ds["input_ids"], dtype=torch.long)
            assert ids.shape == (n, SEQLEN), ids.shape
            out = Path("data/calib") / f"c4_{model}_{n}x{SEQLEN}_seed0.pt"
            torch.save(ids, out)
            sets[n] = ids
            print(f"{model}: {tuple(ids.shape)} -> {out}")
        same = torch.equal(sets[800][:128], sets[128])
        print(f"{model}: first 128 of the 800 windows = the 128 GPTQ windows: {same}")
        assert same


if __name__ == "__main__":
    main()
