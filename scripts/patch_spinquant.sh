#!/usr/bin/env bash
# Our changes to the official SpinQuant code in third_party/SpinQuant (listed in docs/experimental_setup.md).
# Safe to run more than once. `git -C third_party/SpinQuant diff` shows every change.
# Run from ~/Q_experiments:  bash scripts/patch_spinquant.sh
set -e -o pipefail
cd third_party/SpinQuant

# Change 1: gloo instead of NCCL for torch.distributed.
#   NCCL 2.26 (shipped with PyTorch 2.7) cannot start on the A40-24Q vGPU: cudaMemPoolCreate fails with
#   "operation not supported" (NCCL init.cc:416), and no NCCL setting skips it. SpinQuant starts a process group even
#   on one GPU, but with one process nothing is exchanged: the code only asks for the rank and waits at barriers.
#   gloo (PyTorch's CPU backend) does the same and cannot change any result.
for f in ptq.py optimize_rotation.py; do
  sed -i 's/dist.init_process_group(backend="nccl"/dist.init_process_group(backend="gloo"/' "$f"
  grep -q 'init_process_group(backend="gloo"' "$f" || { echo "patch failed in $f"; exit 1; }
done

# Change 2: our C4 calibration windows instead of WikiText-2 train (fairness rule 1: never calibrate on the corpus
#   we evaluate on). Only active when Q_CALIB_FILE points to a file from scripts/export_calib.py; without it the
#   code behaves exactly as the original.
#   - utils/data_utils.py get_wikitext2(eval_mode=False): the GPTQ calibration samples
#   - optimize_rotation.py: the rotation-training samples
# Change 3: Q_ROTATION_SEED seeds the random initial rotation of SpinQuant's training (the original never seeds it).
python - <<'EOF'
from pathlib import Path


def patch(path, old, new, marker):
    text = Path(path).read_text()
    if marker in text:
        return print(f"{path}: already patched ({marker})")
    assert text.count(old) == 1, f"{path}: expected text not found exactly once, the code differs from commit 8f47aa3"
    Path(path).write_text(text.replace(old, new))
    print(f"{path}: patched ({marker})")


patch(
    "utils/data_utils.py",
    "import random\n",
    "import os\nimport random\n",
    "import os\n",
)
patch(
    "utils/data_utils.py",
    '''    else:
        traindata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "train"
        ]''',
    '''    elif os.environ.get("Q_CALIB_FILE"):
        # Our change 2 (Q_experiments/scripts/patch_spinquant.sh): our C4 windows instead of WikiText-2 train.
        ids = torch.load(os.environ["Q_CALIB_FILE"])[:nsamples, :seqlen]
        assert ids.shape == (nsamples, seqlen), ids.shape
        trainloader = []
        for row in ids:
            inp = row[None, :].clone()
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        print(f"Calibration data: {nsamples} x {seqlen} tokens from {os.environ['Q_CALIB_FILE']}")
        return trainloader
    else:
        traindata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "train"
        ]''',
    "Q_CALIB_FILE",
)
patch(
    "optimize_rotation.py",
    "    trainable_parameters = [model.R1.weight] + [",
    '''    if os.environ.get("Q_CALIB_FILE"):
        # Our change 2 (Q_experiments/scripts/patch_spinquant.sh): our C4 windows instead of WikiText-2 train.
        ids = torch.load(os.environ["Q_CALIB_FILE"])
        train_data = [dict(input_ids=row, labels=row.clone()) for row in ids]
        log.info("Calibration data: {} x {} tokens from {}".format(*ids.shape, os.environ["Q_CALIB_FILE"]))

    trainable_parameters = [model.R1.weight] + [''',
    "Q_CALIB_FILE",
)
patch(
    "optimize_rotation.py",
    '    R1 = random_hadamard_matrix(model.config.hidden_size, "cuda")',
    '''    if os.environ.get("Q_ROTATION_SEED"):
        # Our change 3 (Q_experiments/scripts/patch_spinquant.sh): seed the random initial rotation.
        torch.manual_seed(int(os.environ["Q_ROTATION_SEED"]))
    R1 = random_hadamard_matrix(model.config.hidden_size, "cuda")''',
    "Q_ROTATION_SEED",
)
EOF

echo "SpinQuant changes vs commit $(git rev-parse --short HEAD):"
git diff --stat
