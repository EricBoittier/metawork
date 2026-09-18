"""Aggregate PET-MAD fine-tuning cells into a report.

Same idea as check_resume.py: reads every `results/pet_mad_cells/**/*.json`
produced by run_pet_mad_cell.py, and reports throughput and
best_val_metric per (variant, dataset, workers), plus the vs-baseline
comparison on both. Always exits 0 and writes its report regardless of
what it found — a crash is disclosed in the table, not swallowed by a
deleted Snakemake output.
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


def fmt(value: Optional[float], digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", default="baseline")
    args = parser.parse_args()

    cells = load_cells(args.cells)
    by_key: Dict[tuple, Dict[str, Any]] = {}
    for cell in cells:
        key = (cell["dataset"], cell["num_workers"], cell["variant"])
        by_key[key] = cell

    lines = [
        "# PET-MAD fine-tuning benchmark",
        "",
        "Fine-tunes a pretrained PET-MAD checkpoint (not a small model "
        "trained from scratch) on the same datasets `results/summary.md` "
        "uses, with the same per-stage timing. Tests whether the "
        "data-loading changes this harness benchmarks help or hurt at "
        "PET-MAD's much larger model size (102 atomic types, energy + "
        "non-conservative force + stress heads) and richer per-batch "
        "compute, compared to the small energy-only model everything else "
        "in this harness trains from scratch.",
        "",
        "| dataset | workers | variant | status | atoms/s | vs baseline | "
        "best_val_metric | vs baseline |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    crashed = 0
    groups = sorted({(c["dataset"], c["num_workers"]) for c in cells})
    for dataset, workers in groups:
        baseline_cell = by_key.get((dataset, workers, args.baseline))
        baseline_atoms = None
        baseline_metric = None
        if baseline_cell and baseline_cell.get("status") == "ok":
            baseline_atoms = baseline_cell.get("throughput", {}).get("atoms/s")
            baseline_metric = baseline_cell.get("header", {}).get("best_val_metric")

        for key, cell in sorted(by_key.items()):
            if key[0] != dataset or key[1] != workers:
                continue
            variant = key[2]
            status = cell.get("status")
            if status != "ok":
                if status in ("crashed",):
                    crashed += 1
                lines.append(
                    f"| {dataset} | {workers} | {variant} | "
                    f"**{status.upper()}** | — | — | — | — |"
                )
                continue

            atoms = cell.get("throughput", {}).get("atoms/s")
            metric = cell.get("header", {}).get("best_val_metric")
            atoms_speedup = (
                None
                if atoms is None or not baseline_atoms
                else atoms / baseline_atoms
            )
            metric_diff = (
                None
                if metric is None or baseline_metric is None
                else relative_diff(metric, baseline_metric)
            )
            lines.append(
                "| {dataset} | {workers} | {variant} | ok | {atoms} | {speedup} | "
                "{metric} | {diff} |".format(
                    dataset=dataset,
                    workers=workers,
                    variant=variant,
                    atoms=fmt(atoms, 0),
                    speedup="ref" if variant == args.baseline else fmt(atoms_speedup, 2),
                    metric=fmt(metric, 4),
                    diff="ref" if variant == args.baseline else (
                        "—" if metric_diff is None else f"{metric_diff:.0%}"
                    ),
                )
            )

    lines.append("")
    if crashed:
        lines.append(
            f"**{crashed} cell(s) crashed** — see their `log_tail` in the "
            "matching `results/pet_mad_cells/**/*.json`."
        )
    else:
        lines.append("No cell crashed.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")

    if crashed:
        sys.stderr.write(f"check_pet_mad: {crashed} cell(s) crashed — see {args.out}\n")


if __name__ == "__main__":
    main()
