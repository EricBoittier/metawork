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
    "world_size",
    "host",
    "gpu",
    "torch",
    "n_train",
    "n_val",
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
        "world_size": cell.get("world_size", 1),
        "host": cell.get("host"),
        "gpu": env.get("gpu"),
        "torch": env.get("torch"),
        "n_train": header.get("n_train"),
        "n_val": header.get("n_val"),
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
            row.get("world_size", 1),
        )
        groups[key].append(row)

    medians = []
    for key, members in groups.items():
        dataset, workers, batch, device, variant, world_size = key
        medians.append(
            {
                "dataset": dataset,
                "num_workers": workers,
                "batch_size": batch,
                "device": device,
                "variant": variant,
                "world_size": world_size,
                "n": len(members),
                "atoms_per_s": median(m["atoms_per_s"] for m in members),
                "loader_ms": median(m["loader_ms"] for m in members),
                "step_ms": median(m["step_ms"] for m in members),
                "unpack_ms": median(m["unpack_ms"] for m in members),
                "h2d_ms": median(m["h2d_ms"] for m in members),
                "serialize_ms": median(m["serialize_ms"] for m in members),
                "peak_gb": median(m["peak_gb"] for m in members),
            }
        )

    baselines = {
        (m["dataset"], m["num_workers"], m["batch_size"], m["device"], m["world_size"]): m[
            "atoms_per_s"
        ]
        for m in medians
        if m["variant"] == baseline
    }
    # world_size only varies for the distributed sweep; every main-matrix row
    # has world_size 1, so show the column only when something is >1.
    show_world_size = any(m["world_size"] != 1 for m in medians)

    lines = [
        "# Pipeline benchmark summary",
        "",
        f"Median over repeats. Speedup is vs `{baseline}` on the same "
        "dataset / workers / batch / device"
        + (" / world_size" if show_world_size else "") + ".",
        "",
        "| dataset | workers | "
        + ("world_size | " if show_world_size else "")
        + "variant | n | atoms/s | vs baseline | "
        "loader ms | step ms | unpack ms | h2d ms | serialize ms | peak GB |",
        "| --- | ---: | "
        + ("---: | " if show_world_size else "")
        + "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = sorted(
        medians,
        key=lambda m: (
            m["dataset"],
            m["num_workers"],
            m["world_size"],
            m["variant"] != baseline,
            m["variant"],
        ),
    )
    for row in order:
        key = (
            row["dataset"],
            row["num_workers"],
            row["batch_size"],
            row["device"],
            row["world_size"],
        )
        base = baselines.get(key)
        speedup = (
            None
            if row["atoms_per_s"] is None or not base
            else row["atoms_per_s"] / base
        )
        row_template = (
            "| {dataset} | {workers} | "
            + ("{world_size} | " if show_world_size else "")
            + "{variant} | {n} | {atoms} | {vs} | "
            "{loader} | {step} | {unpack} | {h2d} | {serialize} | {peak} |"
        )
        lines.append(
            row_template.format(
                dataset=row["dataset"],
                workers=row["num_workers"],
                world_size=row["world_size"],
                variant=row["variant"],
                n=row["n"],
                atoms=fmt(row["atoms_per_s"], 0),
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
