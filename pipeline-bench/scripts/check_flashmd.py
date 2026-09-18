"""Aggregate FlashMD cells into a report.

Same idea as check_pet_mad.py, but FlashMD's trainer has no
metatrain.utils.timing instrumentation, so there's no atoms/s to compare —
`structures_per_s` here is a coarse derived figure (total structures
processed across all epochs / wall time), not the fine-grained,
GPU-sync-aware throughput the PET-based benchmarks report. Good enough to
compare variants against each other, not to compare against
`results/summary.md`'s numbers.

Always exits 0 and writes its report regardless of what it found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_cells(paths: List[Path]) -> List[Dict[str, Any]]:
    return [json.loads(p.read_text()) for p in paths]


def relative_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def fmt(value: Optional[float], digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", default="baseline")
    args = parser.parse_args()

    cells = load_cells(args.cells)
    by_workers: Dict[int, Dict[str, Any]] = {}
    for cell in cells:
        by_workers.setdefault(cell["num_workers"], {})[cell["variant"]] = cell

    lines = [
        "# FlashMD benchmark",
        "",
        "FlashMD's trainer has no per-stage timing instrumentation (unlike "
        "PET's), and none of the perf/data-loading branches this harness "
        "compares touch its code at all — confirmed by diff against the "
        "stacked PRs. So a difference between variants here cannot be "
        "attributed to those changes; it says something about the "
        "variant's branch/environment more generally.",
        "",
        "| workers | variant | status | wall_s | structures/s | "
        "best_val_metric | vs baseline |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    crashed = 0
    for workers in sorted(by_workers):
        by_variant = by_workers[workers]
        baseline_cell = by_variant.get(args.baseline)
        baseline_metric = None
        if baseline_cell and baseline_cell.get("status") == "ok":
            baseline_metric = baseline_cell.get("header", {}).get("best_val_metric")

        for variant in sorted(by_variant):
            cell = by_variant[variant]
            status = cell.get("status")
            if status != "ok":
                if status == "crashed":
                    crashed += 1
                lines.append(
                    f"| {workers} | {variant} | **{status.upper()}** | — | — | — | — |"
                )
                continue

            header = cell.get("header", {})
            wall_s = header.get("wall_s")
            n_total = (header.get("n_train") or 0) + (header.get("n_val") or 0)
            epochs = header.get("epochs") or 0
            structures_per_s = (
                None if not wall_s else n_total * epochs / wall_s
            )
            metric = header.get("best_val_metric")
            diff = (
                None
                if metric is None or baseline_metric is None
                else relative_diff(metric, baseline_metric)
            )
            lines.append(
                "| {workers} | {variant} | ok | {wall} | {sps} | {metric} | {diff} |".format(
                    workers=workers,
                    variant=variant,
                    wall=fmt(wall_s, 1),
                    sps=fmt(structures_per_s, 1),
                    metric=fmt(metric, 6),
                    diff="ref" if variant == args.baseline else (
                        "—" if diff is None else f"{diff:.0%}"
                    ),
                )
            )

    lines.append("")
    if crashed:
        lines.append(
            f"**{crashed} cell(s) crashed** — see their `log_tail` in the "
            "matching `results/flashmd_cells/**/*.json`."
        )
    else:
        lines.append("No cell crashed.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")

    if crashed:
        sys.stderr.write(f"check_flashmd: {crashed} cell(s) crashed — see {args.out}\n")


if __name__ == "__main__":
    main()
