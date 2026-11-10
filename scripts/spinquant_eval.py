"""Group C driver for SpinQuant's code: one run = quantize + perplexity (+ zero-shot) + cost, saved as JSON.

Builds the model exactly as SpinQuant's ptq.py does (same functions, same order: load, untie the embeddings,
ptq_model = rotate + quantize), then measures:
  - WikiText-2 perplexity with SpinQuant's own evaluator (the numbers ptq.py prints)
  - with --zeroshot: the six zero-shot tasks with our zeroshot_eval.evaluate_zeroshot (lm-eval, batch 16),
    the same code as groups A and B
  - quantization time (ptq_model: rotation + GPTQ/RTN) and peak GPU memory
Everything after "--" is passed to SpinQuant's argument parser unchanged.

Usage (from ~/Q_experiments, spinquant venv, patch_spinquant.sh applied):
  python scripts/spinquant_eval.py --name Llama-3.2-1B_spinquant_QuaRot-GPTQ_W4A4KV4_seed0 \
      --calib data/calib/c4_Llama-3.2-1B_128x2048_seed0.pt -- --input_model meta-llama/Llama-3.2-1B ...
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPIN = ROOT / "third_party" / "SpinQuant"

ours = argparse.ArgumentParser()
ours.add_argument("--name", required=True)
ours.add_argument("--calib", default=None, help="token-ID file from export_calib.py (GPTQ calibration)")
ours.add_argument("--zeroshot", action="store_true")
ours.add_argument("--batch-size", type=int, default=16, help="zero-shot batch size (16, as in groups A and B)")
ours.add_argument("--out-dir", default=str(ROOT / "results" / "groupC"))
ours.add_argument("--extra", default=None, help="JSON file merged into the result (e.g. SpinQuant's rotation-training cost)")
args, rest = ours.parse_known_args()
spin_args = rest[1:] if rest and rest[0] == "--" else rest
if args.calib:
    os.environ["Q_CALIB_FILE"] = str(Path(args.calib).resolve())
# One process, no torchrun: SpinQuant still calls torch.distributed, so give it a one-process group.
for key, value in {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29511", "RANK": "0", "WORLD_SIZE": "1"}.items():
    os.environ.setdefault(key, value)

sys.argv = [sys.argv[0]] + spin_args
sys.path.insert(0, str(SPIN))
sys.path.append(str(ROOT / "scripts"))
os.chdir(SPIN)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import transformers  # noqa: E402
from eval_utils.main import ptq_model  # noqa: E402
from eval_utils.modeling_llama import LlamaForCausalLM  # noqa: E402
from transformers import LlamaTokenizerFast  # noqa: E402
from utils import data_utils, eval_utils, utils  # noqa: E402
from utils.process_args import process_args_ptq  # noqa: E402


def git(*cmd):
    return subprocess.run(["git", "-C", str(SPIN), *cmd], capture_output=True, text=True).stdout.strip()


def main():
    assert "Q_CALIB_FILE" not in os.environ or Path(os.environ["Q_CALIB_FILE"]).exists(), "calibration file missing"
    start = time.perf_counter()
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    if ptq_args.w_bits < 16 and not ptq_args.w_rtn and "Q_CALIB_FILE" not in os.environ:
        sys.exit("GPTQ needs --calib (otherwise SpinQuant would calibrate on WikiText-2 train)")

    # --- as in SpinQuant's ptq.py ---
    config = transformers.AutoConfig.from_pretrained(model_args.input_model, token=model_args.access_token)
    process_word_embeddings = False
    if config.tie_word_embeddings:  # Llama-3.2: untie, so embedding and lm_head can be rotated separately
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model, config=config, torch_dtype=dtype, token=model_args.access_token
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()

    torch.cuda.synchronize()
    gpu_before = torch.cuda.memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = ptq_model(ptq_args, model, model_args)
    torch.cuda.synchronize()
    quant_seconds = time.perf_counter() - t0
    quant_peak = torch.cuda.max_memory_allocated() / 2**30

    model.seqlen = training_args.model_max_length
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    model.config.use_cache = False
    testloader = data_utils.get_wikitext2(seed=ptq_args.seed, seqlen=2048, tokenizer=tokenizer, eval_mode=True)
    t1 = time.perf_counter()
    ppl = float(eval_utils.evaluator(model, testloader, utils.DEV, ptq_args))
    ppl_seconds = time.perf_counter() - t1
    print(f"wiki2 ppl is: {ppl}")
    # --- end of ptq.py ---

    result = {
        "name": args.name,
        "codebase": "spinquant",
        "commit": git("rev-parse", "HEAD"),
        "local_changes": git("diff", "--stat").splitlines()[-1:] or None,
        "args": " ".join(spin_args),
        "calibration": os.environ.get("Q_CALIB_FILE"),
        "ppl": ppl,
        "quant_seconds": round(quant_seconds, 1),
        "quant_peak_gpu_gib": round(quant_peak, 2),
        "gpu_before_quant_gib": round(gpu_before, 2),  # the BF16 model itself: ptq.py moves it to the GPU first
        "ppl_eval_seconds": round(ppl_seconds, 1),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }

    if args.zeroshot:
        from zeroshot_eval import evaluate_zeroshot  # our scripts/zeroshot_eval.py, as in groups A and B

        model.cuda()  # SpinQuant's evaluator leaves the layers on the CPU
        model.config.use_cache = False
        t2 = time.perf_counter()
        scores, average = evaluate_zeroshot(model, tokenizer, args.batch_size)
        result.update({
            "zeroshot_tasks": scores,
            "zeroshot_average": average,
            "zeroshot_batch_size": args.batch_size,
            "zeroshot_seconds": round(time.perf_counter() - t2, 1),
        })
        print(f"zero-shot average: {average}")

    if args.extra:
        result.update(json.loads(Path(args.extra).read_text()))
    result["wall_time_s"] = round(time.perf_counter() - start, 1)
    out = Path(args.out_dir) / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"{args.name}: PPL {ppl:.3f} | quantization {quant_seconds:.0f} s, peak GPU {quant_peak:.2f} GiB -> {out}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
