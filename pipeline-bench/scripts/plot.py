"""Plot atoms/s vs workers, one panel per dataset."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def load_cells(path: Path) -> List[Dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is not installed in this environment")

    rows = load_cells(args.cells_csv)
    grouped: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    for row in rows:
        if not row.get("atoms_per_s"):
            continue
        key = (row["dataset"], row["variant"], int(row["num_workers"]))
        grouped[key].append(float(row["atoms_per_s"]))

    datasets = sorted({dataset for dataset, _, _ in grouped})
    if not datasets:
        raise SystemExit(f"no atoms/s values in {args.cells_csv}")

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(5.2 * len(datasets), 4.2), squeeze=False
    )
    for ax, dataset in zip(axes[0], datasets):
        variants = sorted({v for d, v, _ in grouped if d == dataset})
        for variant in variants:
            workers = sorted({w for d, v, w in grouped if d == dataset and v == variant})
            ys = [
                median(grouped[(dataset, variant, w)]) for w in workers
            ]
            ax.plot(workers, ys, marker="o", label=variant)
        ax.set_title(f"atoms/s · {dataset}")
        ax.set_xlabel("num_workers")
        ax.set_ylabel("atoms / s (median over repeats)")
        ax.legend(frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
