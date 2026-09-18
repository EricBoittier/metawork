"""Check whether run order correlates with measured speed.

Snakemake's scheduler doesn't randomize job order against variant identity —
with `--cores 1` it tends to run one variant's cells as a contiguous block
right after that variant's worktree is built, rather than interleaving across
variants. Any thermal throttling or background load that drifts over the
course of a multi-minute run would then be confounded with *which variant
ran when*, not just with the mechanism under test.

This can't force randomized interleaving (that would need surgery on
Snakemake's scheduler), so instead it detects and discloses: for every cell,
normalize `atoms_per_s` against the median for its own (dataset, workers,
batch, device) group — this removes the "some configs are just faster"
effect — then check whether that normalized value trends with chronological
run order, both globally and within each variant's own block.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load(cells_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cells_csv, parse_dates=["started"])
    df = df.dropna(subset=["started", "atoms_per_s"])
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "num_workers", "batch_size", "device"]
    medians = df.groupby(group_cols)["atoms_per_s"].median().rename("group_median")
    df = df.join(medians, on=group_cols)
    df["relative"] = df["atoms_per_s"] / df["group_median"]
    return df


def spearman(order: pd.Series, value: pd.Series) -> float | None:
    """Spearman rho as the Pearson correlation of ranks — avoids a scipy dep."""
    if len(order) < 4:
        return None
    return float(order.rank().corr(value.rank()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--flag-threshold",
        type=float,
        default=0.3,
        help="absolute Spearman rho above which drift is flagged as notable",
    )
    args = parser.parse_args()

    df = load(args.cells_csv)
    df = normalize(df).sort_values("started").reset_index(drop=True)
    df["run_order"] = range(len(df))

    lines = ["# Run-order drift check", ""]
    lines.append(
        "`relative` = atoms/s divided by the median atoms/s for that exact "
        "(dataset, workers, batch, device) combination — 1.0 means typical "
        "for its own config, so this is comparable across configs and variants."
    )
    lines.append("")

    global_rho = spearman(df["run_order"], df["relative"])
    lines.append(
        f"**Global** (all {len(df)} cells, chronological): "
        f"Spearman rho = {global_rho:.2f}"
        if global_rho is not None
        else "**Global**: not enough cells to test."
    )
    if global_rho is not None and abs(global_rho) >= args.flag_threshold:
        direction = "slower" if global_rho < 0 else "faster"
        lines.append(
            f"  - **FLAGGED**: cells run later in the matrix trend {direction} "
            "relative to their own config's median, independent of variant. "
            "That's consistent with thermal drift, background load, or a "
            "warming/cooling GPU over the run — treat variant-vs-baseline "
            "deltas for whichever variants ran at the extremes of the run "
            "with extra skepticism."
        )
    lines.append("")

    lines.append("**Per-variant** (order within that variant's own block):")
    for variant, group in df.groupby("variant"):
        group = group.sort_values("started").reset_index(drop=True)
        rho = spearman(pd.Series(range(len(group))), group["relative"])
        if rho is None:
            lines.append(f"- `{variant}`: not enough cells to test.")
            continue
        flag = " — **FLAGGED**" if abs(rho) >= args.flag_threshold else ""
        lines.append(f"- `{variant}`: rho = {rho:.2f}, n = {len(group)}{flag}")

    lines.append("")
    lines.append(
        "A flagged rho is a correlation, not a diagnosis — it says the data "
        "is consistent with an order effect, not that one is proven. Rerunning "
        "the flagged variant's cells interleaved with another variant's (or "
        "just rerunning the matrix and checking whether the flag persists) is "
        "the actual follow-up, not adjusting the numbers based on this alone."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
