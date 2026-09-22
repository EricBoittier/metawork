"""Profile one variant's PET training run with torch.profiler.

Wraps the whole ``benchmark_pipeline.main()`` call (dataset build, model
build, ``Trainer.train()``) in a ``torch.profiler`` session so the per-op
CPU/CUDA breakdown is visible on top of the harness's own coarse stage
timers -- the manual ``loader``/``step`` timers in ``benchmark_pipeline.py``
say *which stage* is slow; this says *which kernels/ops* inside that stage
actually spend the time.

Run as a fresh subprocess per variant (like run_cell.py's cells), since two
variants' ``metatrain.*`` packages share the same dotted import path at
different filesystem locations -- importing both in one process would let
whichever loads first shadow the other.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-prefix", required=True, help="e.g. results/profile/everything_qm9")
    args = parser.parse_args()

    sys.path.insert(0, str(args.worktree / "src"))
    benchmark_script = args.worktree / "benchmarks" / "benchmark_pipeline.py"

    sys.argv = [
        str(benchmark_script),
        "--dataset", args.dataset,
        "--key", args.key,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--epochs", str(args.epochs),
        "--device", args.device,
    ]

    activities = [ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        # benchmark_pipeline.py is a script (module-level `if __name__ ==
        # "__main__": main()`), not an importable package -- run it as
        # __main__ under the profiler instead of duplicating its setup code.
        runpy.run_path(str(benchmark_script), run_name="__main__")

    prof.export_chrome_trace(str(out_prefix) + ".trace.json")

    sort_keys = ["self_cuda_time_total", "self_cpu_time_total"] if args.device == "cuda" else ["self_cpu_time_total"]
    with open(str(out_prefix) + ".table.txt", "w") as f:
        for key in sort_keys:
            f.write(f"=== sorted by {key} ===\n")
            f.write(prof.key_averages().table(sort_by=key, row_limit=40))
            f.write("\n\n")

    print(f"wrote {out_prefix}.trace.json and {out_prefix}.table.txt")


if __name__ == "__main__":
    main()
