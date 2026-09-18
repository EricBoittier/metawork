"""Run one pipeline-benchmark cell against a variant worktree."""

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


def probe(python: Path) -> dict:
    code = (
        "import json,sys,torch;"
        "gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None;"
        "print(json.dumps({"
        "'python': sys.version.split()[0],"
        "'torch': torch.__version__,"
        "'cuda': torch.cuda.is_available(),"
        "'gpu': gpu,"
        "}))"
    )
    result = subprocess.run(
        [str(python), "-c", code], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)


def run_benchmark(
    python: Path,
    worktree: Path,
    dataset: Path,
    key: str,
    num_workers: int,
    batch_size: int,
    device: str,
    epochs: int,
    timeout_s: int,
    log_path: Path,
    model_hypers_json: str | None = None,
    max_atoms_per_batch: int | None = None,
    world_size: int = 1,
) -> str:
    script = worktree / "benchmarks" / "benchmark_pipeline.py"
    if not script.is_file():
        raise SystemExit(f"no benchmark script at {script}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(worktree / "src")
    env["METATRAIN_TIMING"] = "1"
    cmd = [
        str(python),
        str(script),
        "--dataset",
        str(dataset),
        "--key",
        key,
        "--num-workers",
        str(num_workers),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--epochs",
        str(epochs),
    ]
    if model_hypers_json:
        cmd += ["--model-hypers-json", model_hypers_json]
    if max_atoms_per_batch:
        cmd += ["--max-atoms-per-batch", str(max_atoms_per_batch)]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        # metatrain's Trainer.train() auto-enables DistributedDataParallel
        # whenever SLURM_NTASKS > 1 (see
        # metatrain.utils.distributed.slurm.resolve_distributed) and does its
        # own process-group setup from the SLURM_* env -- no extra flags
        # needed here beyond launching one task per rank. Each rank's stdout
        # goes to its own file (srun's %t), since interleaving N ranks'
        # output into one pipe would garble it; rank 0's becomes the log
        # that parse_report() reads, matching the single-GPU cell's output.
        rank_stem = log_path.with_suffix("")
        full_cmd = [
            "srun",
            "--ntasks",
            str(world_size),
            "--output",
            f"{rank_stem}.rank%t.log",
        ] + cmd
        log_path.write_text("# " + " ".join(full_cmd) + "\n")
        proc = subprocess.run(
            full_cmd, env=env, timeout=timeout_s, check=False
        )
        rank0_log = Path(f"{rank_stem}.rank0.log")
        text = rank0_log.read_text() if rank0_log.is_file() else ""
        with log_path.open("a") as log:
            log.write(text)
    else:
        with log_path.open("w") as log:
            log.write("# " + " ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.run(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
        text = log_path.read_text()

    if proc.returncode != 0:
        raise SystemExit(
            f"benchmark exited {proc.returncode}; see {log_path}\n{text[-2000:]}"
        )
    return text


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
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--variant", default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--model-hypers-json",
        default=None,
        help="JSON dict merged into the default PET model hypers, e.g. to "
        "test a larger model size",
    )
    parser.add_argument(
        "--max-atoms-per-batch",
        type=int,
        default=None,
        help="pack batches under this total atom count instead of a fixed "
        "--batch-size structure count",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="number of ranks/GPUs; >1 launches via `srun --ntasks` for "
        "real DistributedDataParallel (metatrain auto-enables it from "
        "SLURM_NTASKS)",
    )
    args = parser.parse_args()

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stdout = run_benchmark(
        args.python,
        args.worktree,
        args.dataset,
        args.key,
        args.num_workers,
        args.batch_size,
        args.device,
        args.epochs,
        args.timeout_s,
        args.log,
        args.model_hypers_json,
        args.max_atoms_per_batch,
        args.world_size,
    )
    parsed = parse_report(stdout)
    sha_file = args.worktree / ".variant-shas"
    cell = {
        "variant": args.variant,
        "dataset": args.dataset_name or args.dataset.name,
        "repeat": args.repeat,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "device": args.device,
        "epochs": args.epochs,
        "world_size": args.world_size,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "worktree": str(args.worktree),
        "shas": sha_file.read_text().split() if sha_file.is_file() else [],
        "env": probe(args.python),
        **parsed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cell, indent=2) + "\n")


if __name__ == "__main__":
    main()
