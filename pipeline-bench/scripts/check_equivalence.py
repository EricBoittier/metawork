"""Sanity-check that variants sharing a dataset actually trained on the same data.

This is *not* a correctness check on model outputs — nothing here compares
loss, gradients, or predictions across variants. Each branch's
``benchmarks/benchmark_pipeline.py`` is a separate script, so a change that
silently drops structures, double-counts a batch, or otherwise changes the
workload would still need a real loss/gradient comparison to catch. What
this script *can* catch cheaply, from data pipeline-bench already collects,
is the coarser bug: a variant whose ``n_train``/``n_val`` for a given
dataset don't match its siblings, which either means the workload isn't
actually equivalent (e.g. the non-disjoint validation split fixed in
metatrain's `fix/pipeline-benchmark-val-split`) or means the branch's report
format doesn't match `report.py`'s parser closely enough to tell.

Exits non-zero if any dataset has disagreeing sizes, so it can gate `rule all`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_rows(cells_csv: Path) -> List[Dict[str, str]]:
    with cells_csv.open(newline="") as handle:
        return list(csv.DictReader(handle))


def check(rows: List[Dict[str, str]]) -> List[str]:
    by_dataset: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    missing: Dict[str, List[str]] = defaultdict(list)

    for row in rows:
        dataset = row["dataset"]
        variant = row["variant"]
        n_train, n_val = row.get("n_train"), row.get("n_val")
        if not n_train or not n_val:
            missing[dataset].append(variant)
            continue
        by_dataset[dataset]["n_train"].add(n_train)
        by_dataset[dataset]["n_val"].add(n_val)
        by_dataset[dataset].setdefault("sizes", set()).add((variant, n_train, n_val))

    lines = ["# Equivalence check", ""]
    problems = 0
    for dataset in sorted(set(by_dataset) | set(missing)):
        sizes = by_dataset.get(dataset, {})
        n_train_vals = sizes.get("n_train", set())
        n_val_vals = sizes.get("n_val", set())
        missing_variants = sorted(set(missing.get(dataset, [])))

        if len(n_train_vals) > 1 or len(n_val_vals) > 1:
            problems += 1
            lines.append(
                f"- **FAIL** `{dataset}`: variants disagree on split size — "
                f"n_train in {sorted(n_train_vals)}, n_val in {sorted(n_val_vals)}. "
                "These variants are not training on equivalent data; any "
                "speed comparison between them is not apples-to-apples."
            )
        elif n_train_vals:
            lines.append(
                f"- **OK** `{dataset}`: all reporting variants agree on "
                f"n_train={next(iter(n_train_vals))}, n_val={next(iter(n_val_vals))}."
            )
        if missing_variants:
            lines.append(
                f"  - **UNVERIFIED** for `{dataset}`: {', '.join(missing_variants)} "
                "did not report n_train/n_val at all, so they can't be checked "
                "against the others — treat their numbers as unverified, not as passing."
            )

    lines.append("")
    lines.append(
        "This checks workload *shape* only (dataset sizes), never model "
        "correctness. No loss, gradient, or output comparison exists between "
        "variants in this harness — that remains open follow-up work."
    )
    return lines, problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.cells_csv)
    lines, problems = check(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")

    if problems:
        sys.stderr.write(
            f"check_equivalence: {problems} dataset(s) with disagreeing split "
            f"sizes — see {args.out}\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
