"""Save one group C run as results/groupC/<name>.json, reading the perplexity from the run's log.

Each research codebase prints its result differently, so --pattern is a regex with one group for the perplexity.
Default: SpinQuant's ptq.py ("wiki2 ppl is: 9.76...").
Called by the group C run scripts; it exits with an error if the log has no perplexity (a failed run).
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--codebase", default="spinquant")
    parser.add_argument("--code-dir", default="third_party/SpinQuant", help="git folder of the codebase (for the commit)")
    parser.add_argument("--seconds", type=int, required=True, help="wall time of the whole run (load, quantize, evaluate)")
    parser.add_argument("--args", default="", help="the codebase's command-line arguments")
    parser.add_argument("--pattern", default=r"wiki2 ppl is: ([0-9.]+)")
    args = parser.parse_args()

    found = re.findall(args.pattern, Path(args.log).read_text(errors="ignore"))
    if not found:
        sys.exit(f"No perplexity in {args.log}: the run failed. Look at the end of the log.")
    try:
        commit = subprocess.run(["git", "-C", args.code_dir, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", args.code_dir, "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = None, None
    result = {
        "name": args.name,
        "codebase": args.codebase,
        "commit": commit,
        "local_changes": dirty,  # True = our patches are applied (scripts/patch_*.sh)
        "ppl": float(found[-1]),
        "wall_time_s": args.seconds,
        "args": args.args,
        "log": args.log,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"{args.name}: PPL {result['ppl']:.3f} ({args.seconds} s) -> {out}")


if __name__ == "__main__":
    main()
