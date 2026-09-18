"""Aggregate resume-check cells into a report.

Reads every `results/resume_cells/**/*.json` produced by
`run_resume_check.py` and reports, per (variant, workers): did checkpoint
save/restart complete at all, how far did the resumed run's
`best_val_metric` drift from the uninterrupted control's, and how much
wall-clock overhead did the checkpoint round trip add.

A crash is the headline finding (a regression this harness can actually
name), not a script error — so like the other check_*.py scripts, this
always exits 0 and writes its report regardless of what it found. Read
`results/resume.md`, don't gate on the exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_cells(paths: List[Path]) -> List[Dict[str, Any]]:
    return [json.loads(p.read_text()) for p in paths]


def relative_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--drift-warn-threshold",
        type=float,
        default=0.20,
        help="relative best_val_metric drift (resumed vs continuous) to flag",
    )
    args = parser.parse_args()

    cells = load_cells(args.cells)
    lines = [
        "# Checkpoint save/resume check",
        "",
        "Per (variant, workers): train straight through vs. train, checkpoint,"
        " restart via the real `mtt train --restart` path, and finish. A"
        " CRASH means resume itself is broken for that variant/worker count —"
        " the actual regression this check exists to catch. A drift flag on a"
        " completed run is expected background noise (the checkpoint doesn't"
        " carry RNG state) unless it's large or one-sided across variants.",
        "",
        "| variant | dataset | workers | status | continuous best | resumed best |"
        " drift | resume overhead (s) |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]

    crashed = 0
    for cell in sorted(cells, key=lambda c: (c["dataset"], c["variant"], c["num_workers"])):
        status = cell.get("status")
        variant = cell["variant"]
        dataset = cell["dataset"]
        workers = cell["num_workers"]
        if status not in ("ok",):
            crashed += 1
            lines.append(
                f"| {variant} | {dataset} | {workers} | **{status.upper()}** "
                "| — | — | — | — |"
            )
            continue
        drift = relative_diff(cell["continuous_best"], cell["resumed_best"])
        flag = " ⚠" if drift >= args.drift_warn_threshold else ""
        lines.append(
            f"| {variant} | {dataset} | {workers} | ok | "
            f"{cell['continuous_best']:.6g} | {cell['resumed_best']:.6g} | "
            f"{drift:.0%}{flag} | {cell['resume_overhead_s']:.2f} |"
        )

    lines.append("")
    if crashed:
        lines.append(
            f"**{crashed} cell(s) did not complete** — see their `log_tail` in "
            "the matching `results/resume_cells/**/*.json` for the traceback."
        )
    else:
        lines.append(
            f"All {len(cells)} cell(s) completed the checkpoint save/restart "
            "round trip without error."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")

    if crashed:
        sys.stderr.write(f"check_resume: {crashed} cell(s) crashed — see {args.out}\n")


if __name__ == "__main__":
    main()
