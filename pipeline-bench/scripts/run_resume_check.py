"""Run the checkpoint save/resume regression check against a variant worktree."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


RESUME_LINE = re.compile(
    r"resume_check status=(?P<status>\S+) split_epoch=(?P<split_epoch>\d+) "
    r"total_epochs=(?P<total_epochs>\d+) "
    r"continuous_best=(?P<continuous_best>[\d.eE+-]+) "
    r"continuous_best_epoch=(?P<continuous_best_epoch>\d+) "
    r"continuous_wall_s=(?P<continuous_wall_s>[\d.]+) "
    r"resumed_best=(?P<resumed_best>[\d.eE+-]+) "
    r"resumed_best_epoch=(?P<resumed_best_epoch>\d+) "
    r"resumed_wall_s=(?P<resumed_wall_s>[\d.]+) "
    r"resume_overhead_s=(?P<resume_overhead_s>[-\d.]+)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--split-epoch", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--variant", default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    script = args.worktree / "benchmarks" / "benchmark_checkpoint_resume.py"
    result = {
        "variant": args.variant,
        "dataset": args.dataset_name or args.dataset.name,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "device": args.device,
        "epochs": args.epochs,
        "split_epoch": args.split_epoch,
        "seed": args.seed,
    }

    if not script.is_file():
        result["status"] = "no_resume_script"
        result["exit_code"] = None
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(f"# no resume script at {script}\n")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.worktree / "src")
    cmd = [
        str(args.python),
        str(script),
        "--dataset",
        str(args.dataset),
        "--key",
        args.key,
        "--num-workers",
        str(args.num_workers),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--split-epoch",
        str(args.split_epoch),
        "--seed",
        str(args.seed),
    ]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w") as log:
        log.write("# " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=args.timeout_s,
            check=False,
        )
    text = args.log.read_text()
    result["exit_code"] = proc.returncode

    match = RESUME_LINE.search(text)
    if match:
        result.update(
            {
                "status": match["status"],
                "continuous_best": float(match["continuous_best"]),
                "continuous_best_epoch": int(match["continuous_best_epoch"]),
                "continuous_wall_s": float(match["continuous_wall_s"]),
                "resumed_best": float(match["resumed_best"]),
                "resumed_best_epoch": int(match["resumed_best_epoch"]),
                "resumed_wall_s": float(match["resumed_wall_s"]),
                "resume_overhead_s": float(match["resume_overhead_s"]),
            }
        )
    else:
        # Deliberately not raising: a crashed resume check *is* the finding.
        # Write it to the JSON so the report can name which variant failed,
        # rather than losing that in a deleted/missing output file.
        result["status"] = "crashed" if proc.returncode != 0 else "unparsed_output"
        result["log_tail"] = text[-2000:]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
