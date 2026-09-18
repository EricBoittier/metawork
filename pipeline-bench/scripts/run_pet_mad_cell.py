"""Run one PET-MAD fine-tuning benchmark cell against a variant worktree.

Mirrors run_cell.py's JSON shape (same report.py parser, so aggregate.py
and the check_*.py scripts work unchanged) but, like run_resume_check.py,
never raises on a missing script or a crash — it writes the failure into
the JSON instead, so a real problem shows up as a finding in
results/pet_mad.md rather than as a deleted Snakemake output.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import parse_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--variant", default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--pet-mad-size", default="")
    parser.add_argument("--pet-mad-version", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cell = {
        "variant": args.variant,
        "dataset": args.dataset_name or args.dataset.name,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "device": args.device,
        "epochs": args.epochs,
        "seed": args.seed,
        "pet_mad_size": args.pet_mad_size,
        "pet_mad_version": args.pet_mad_version,
        "host": socket.gethostname(),
    }

    script = args.worktree / "benchmarks" / "benchmark_pet_mad.py"
    if not script.is_file():
        cell["status"] = "no_pet_mad_script"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(cell, indent=2) + "\n")
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(f"# no PET-MAD benchmark script at {script}\n")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.worktree / "src")
    env["METATRAIN_TIMING"] = "1"
    cmd = [
        str(args.python),
        str(script),
        "--checkpoint",
        str(args.checkpoint),
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

    if proc.returncode != 0:
        cell["status"] = "crashed"
        cell["log_tail"] = text[-2000:]
    else:
        cell["status"] = "ok"
        parsed = parse_report(text)
        cell.update(parsed)
        cell["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cell["started"] = started
        cell["platform"] = platform.platform()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cell, indent=2) + "\n")


if __name__ == "__main__":
    main()
