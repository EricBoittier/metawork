"""Fold per-cell JSON into tidy CSV tables and a markdown summary."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CELL_FIELDS = [
    "variant",
    "dataset",
    "repeat",
    "num_workers",
    "batch_size",
    "device",
    "epochs",
    "seed",
    "host",
    "gpu",
    "torch",
    "n_train",
    "n_val",
    "best_val_metric",
    "best_model_metric",
    "best_epoch",
    "epoch1_train_loss",
    "epoch1_val_loss",
    "wall_s",
    "peak_gb",
    "added_gb",
    "baseline_gb",
    "atoms_per_s",
    "structures_per_s",
    "loader_ms",
    "step_ms",
    "unpack_ms",
    "h2d_ms",
    "serialize_ms",
    "group_and_join_ms",
    "transforms_ms",
    "started",
    "finished",
]


def stage_ms(cell: Dict[str, Any], name: str) -> Optional[float]:
    stage = cell.get("stages", {}).get(name)
    return None if stage is None else stage["ms_per_call"]


def flatten(cell: Dict[str, Any]) -> Dict[str, Any]:
    header = cell.get("header", {})
    env = cell.get("env", {})
    throughput = cell.get("throughput", {})
    return {
        "variant": cell.get("variant"),
        "dataset": cell.get("dataset"),
        "repeat": cell.get("repeat"),
        "num_workers": cell.get("num_workers"),
        "batch_size": cell.get("batch_size"),
        "device": cell.get("device"),
        "epochs": cell.get("epochs"),
        "seed": cell.get("seed"),
        "host": cell.get("host"),
        "gpu": env.get("gpu"),
        "torch": env.get("torch"),
        "n_train": header.get("n_train"),
        "n_val": header.get("n_val"),
        "best_val_metric": header.get("best_val_metric"),
        "best_model_metric": header.get("best_model_metric"),
        "best_epoch": header.get("best_epoch"),
        "epoch1_train_loss": header.get("epoch1_train_loss"),
        "epoch1_val_loss": header.get("epoch1_val_loss"),
        "wall_s": header.get("wall_s"),
        "peak_gb": header.get("peak_gb"),
        "added_gb": header.get("added_gb"),
        "baseline_gb": header.get("baseline_gb"),
        "atoms_per_s": throughput.get("atoms/s"),
        "structures_per_s": throughput.get("structures/s"),
        "loader_ms": stage_ms(cell, "loader"),
        "step_ms": stage_ms(cell, "step"),
        "unpack_ms": stage_ms(cell, "unpack"),
        "h2d_ms": stage_ms(cell, "h2d"),
        "serialize_ms": stage_ms(cell, "serialize"),
        "group_and_join_ms": stage_ms(cell, "group_and_join"),
        "transforms_ms": stage_ms(cell, "transforms"),
        "started": cell.get("started"),
        "finished": cell.get("finished"),
    }


def stage_rows(cell: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for name, stage in sorted(cell.get("stages", {}).items()):
        rows.append(
            {
                "variant": cell.get("variant"),
                "dataset": cell.get("dataset"),
                "repeat": cell.get("repeat"),
                "num_workers": cell.get("num_workers"),
                "batch_size": cell.get("batch_size"),
                "device": cell.get("device"),
                "stage": name,
                "calls": stage["calls"],
                "ms_per_call": stage["ms_per_call"],
                "pct": stage["pct"],
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    return None if not nums else statistics.median(nums)


def spread_pct(values: Iterable[Optional[float]]) -> Optional[float]:
    """(max - min) / median, as a percentage, over non-null repeats.

    A point estimate (the median) hides how much a handful of repeats
    actually varied. This is the cheapest signal for "is this difference
    real or within run-to-run noise" without assuming a distribution.
    """
    nums = [v for v in values if v is not None]
    if len(nums) < 2:
        return None
    mid = statistics.median(nums)
    return None if not mid else (max(nums) - min(nums)) / mid * 100


def fmt(value: Optional[float], digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def summary_markdown(rows: List[Dict[str, Any]], baseline: str) -> str:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["num_workers"],
            row["batch_size"],
            row["device"],
            row["variant"],
        )
        groups[key].append(row)

    medians = []
    for key, members in groups.items():
        dataset, workers, batch, device, variant = key
        medians.append(
            {
                "dataset": dataset,
                "num_workers": workers,
                "batch_size": batch,
                "device": device,
                "variant": variant,
                "n": len(members),
                "atoms_per_s": median(m["atoms_per_s"] for m in members),
                "atoms_per_s_spread": spread_pct(m["atoms_per_s"] for m in members),
                "loader_ms": median(m["loader_ms"] for m in members),
                "step_ms": median(m["step_ms"] for m in members),
                "unpack_ms": median(m["unpack_ms"] for m in members),
                "h2d_ms": median(m["h2d_ms"] for m in members),
                "serialize_ms": median(m["serialize_ms"] for m in members),
                "peak_gb": median(m["peak_gb"] for m in members),
            }
        )

    baselines = {
        (m["dataset"], m["num_workers"], m["batch_size"], m["device"]): m["atoms_per_s"]
        for m in medians
        if m["variant"] == baseline
    }

    epochs_seen = sorted({r["epochs"] for r in rows if r.get("epochs") is not None})
    batches_seen = sorted({r["batch_size"] for r in rows if r.get("batch_size") is not None})
    min_n = min((m["n"] for m in medians), default=0)

    lines = [
        "# Pipeline benchmark summary",
        "",
        f"Median over repeats. Speedup is vs `{baseline}` on the same "
        "dataset / workers / batch / device.",
        "",
        "**Read the spread column before trusting a single-digit-percent "
        "difference.** It's `(max - min) / median` over the repeats in that "
        "cell — a cell with 2 repeats and a wide spread cannot distinguish "
        "a real effect from run-to-run noise.",
        "",
        "Known limitations of this harness (not fixed by more repeats):",
        f"- **n={min_n} repeats minimum** in this run" + (
            " — below 3, spread is a weak signal; treat anything under "
            "~10% difference as unproven." if min_n < 3 else "."
        ),
        "- **Warm-up is not excluded.** The first epoch of each cell "
        "(cuDNN autotune, CUDA context init, allocator/page-lock warmup) "
        "is timed like any other; on short runs this can look like a "
        "per-batch regression that a longer run would amortize away.",
        f"- **Single point in batch-size space**: only batch_size="
        f"{', '.join(map(str, batches_seen))} tested. `pin_memory`'s "
        "benefit scales with transfer size — a variant that loses here "
        "might win at a batch size this sweep never tried.",
        f"- **{epochs_seen[0] if epochs_seen else '?'} epochs per cell** — "
        "mechanisms that amortize a one-time cost across epochs "
        "(e.g. `persistent_workers` avoiding worker respawn) get a "
        "shorter horizon to pay off than a real training run would give them.",
        "- See `results/equivalence.md` for whether variants in the same "
        "dataset actually trained on the same data, `results/correctness.md` "
        "for whether they trained to the same best_val_metric, and "
        "`results/drift.md` for whether run order correlates with the "
        "measured speed.",
        "",
        "| dataset | workers | variant | n | atoms/s | spread | vs baseline | "
        "loader ms | step ms | unpack ms | h2d ms | serialize ms | peak GB |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = sorted(
        medians,
        key=lambda m: (
            m["dataset"],
            m["num_workers"],
            m["variant"] != baseline,
            m["variant"],
        ),
    )
    for row in order:
        key = (row["dataset"], row["num_workers"], row["batch_size"], row["device"])
        base = baselines.get(key)
        speedup = (
            None
            if row["atoms_per_s"] is None or not base
            else row["atoms_per_s"] / base
        )
        lines.append(
            "| {dataset} | {workers} | {variant} | {n} | {atoms} | {spread} | {vs} | "
            "{loader} | {step} | {unpack} | {h2d} | {serialize} | {peak} |".format(
                dataset=row["dataset"],
                workers=row["num_workers"],
                variant=row["variant"],
                n=row["n"],
                atoms=fmt(row["atoms_per_s"], 0),
                spread=(
                    "—" if row["atoms_per_s_spread"] is None
                    else f"{row['atoms_per_s_spread']:.0f}%"
                ),
                vs=fmt(speedup, 2),
                loader=fmt(row["loader_ms"], 2),
                step=fmt(row["step_ms"], 1),
                unpack=fmt(row["unpack_ms"], 2),
                h2d=fmt(row["h2d_ms"], 2),
                serialize=fmt(row["serialize_ms"], 2),
                peak=fmt(row["peak_gb"], 2),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="+", type=Path)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--cells-csv", type=Path, required=True)
    parser.add_argument("--stages-csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    cells = [json.loads(path.read_text()) for path in args.cells]
    flat = [flatten(cell) for cell in cells]
    stages = [row for cell in cells for row in stage_rows(cell)]
    write_csv(args.cells_csv, flat, CELL_FIELDS)
    write_csv(
        args.stages_csv,
        stages,
        [
            "variant",
            "dataset",
            "repeat",
            "num_workers",
            "batch_size",
            "device",
            "stage",
            "calls",
            "ms_per_call",
            "pct",
        ],
    )
    args.summary.write_text(summary_markdown(flat, args.baseline))


if __name__ == "__main__":
    main()
