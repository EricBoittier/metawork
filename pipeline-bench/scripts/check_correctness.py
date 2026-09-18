"""Check that variants train to the same place, not just at the same speed.

Every other check in this harness is about wall-clock speed. This is the
one check that looks at whether the six variants are actually computing the
same thing. It relies on each branch's `benchmarks/benchmark_pipeline.py`:

- seeding `random`/`numpy`/`torch` with the same fixed `SEED` before building
  the dataset or model, so model init, augmentation, and data shuffling draw
  the same sequence of random numbers across variants
- reporting `trainer.best_metric` (the best validation metric seen across
  training, tracked by `Trainer.train` regardless of `log_interval`) as a
  `best_val_metric ... (metric_name) at epoch ...` line

Two things get checked, both from `results/cells.csv`:

1. **Reproducibility**: repeats of the *same* (variant, dataset, workers,
   batch, epochs, device) should give the same `best_val_metric` — the seed
   is fixed, so they're running the identical computation twice. A mismatch
   here means something is nondeterministic that the seeding assumption
   didn't actually cover (e.g. a worker-level RNG not seeded from the base
   seed), and every cross-variant comparison for that cell is suspect.
2. **Cross-variant equivalence**: within a (dataset, workers, batch, epochs,
   device) group, every variant's `best_val_metric` is compared against
   `baseline`'s. A real difference here means two variants are not computing
   the same thing — the exact failure mode no throughput number can catch.

This is still not a full correctness proof — it's one scalar, from one seed,
over a handful of epochs on tiny data. Treat FAIL as "investigate," and OK
as "no problem detected at this resolution," not as "proven correct."
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


GroupKey = Tuple[str, str, str, str, str]


def load_rows(cells_csv: Path) -> List[Dict[str, str]]:
    with cells_csv.open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> Optional[float]:
    return None if not value else float(value)


def relative_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def check_reproducibility(
    rows: List[Dict[str, str]], tolerance: float
) -> Tuple[List[str], int]:
    groups: Dict[Tuple, List[float]] = defaultdict(list)
    for row in rows:
        metric = to_float(row.get("best_val_metric", ""))
        if metric is None:
            continue
        key = (
            row["variant"],
            row["dataset"],
            row["num_workers"],
            row["batch_size"],
            row["epochs"],
            row["device"],
        )
        groups[key].append(metric)

    lines = ["## Reproducibility (same variant, same config, different repeats)", ""]
    problems = 0
    tested = 0
    for key, values in sorted(groups.items()):
        if len(values) < 2:
            continue
        tested += 1
        spread = max(values) - min(values)
        base = max(abs(v) for v in values) or 1e-12
        if spread / base > tolerance:
            problems += 1
            variant, dataset, workers, batch, epochs, device = key
            lines.append(
                f"- **FAIL** `{variant}`/`{dataset}` (w{workers} b{batch} "
                f"e{epochs} {device}): repeats disagree — {sorted(values)}. "
                "The fixed seed didn't make this cell deterministic; treat "
                "any cross-variant comparison for this config as unreliable "
                "until that's understood."
            )
    if tested == 0:
        lines.append(
            "No (variant, config) cell had >=2 repeats with a `best_val_metric` "
            "to compare — nothing was tested here."
        )
    elif problems == 0:
        lines.append(
            f"All {tested} repeated cell(s) reproduced their own "
            f"`best_val_metric` within {tolerance:.0%}."
        )
    return lines, problems


def check_cross_variant(
    rows: List[Dict[str, str]], baseline: str, warn: float, fail: float
) -> Tuple[List[str], int]:
    groups: Dict[GroupKey, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    metric_names: Dict[GroupKey, Dict[str, str]] = defaultdict(dict)
    for row in rows:
        metric = to_float(row.get("best_val_metric", ""))
        if metric is None:
            continue
        key = (row["dataset"], row["num_workers"], row["batch_size"], row["epochs"], row["device"])
        groups[key][row["variant"]].append(metric)
        metric_names[key][row["variant"]] = row.get("best_model_metric", "?")

    lines = [
        "",
        "## Cross-variant equivalence (same config, different variants)",
        "",
        f"vs `{baseline}`, same (dataset, workers, batch, epochs, device). "
        f"WARN >= {warn:.0%} relative difference, FAIL >= {fail:.0%}.",
        "",
    ]
    problems = 0
    for key in sorted(groups):
        dataset, workers, batch, epochs, device = key
        by_variant = groups[key]
        if baseline not in by_variant:
            lines.append(
                f"- `{dataset}` w{workers} b{batch} e{epochs} {device}: "
                f"no `{baseline}` cell to compare against, skipped."
            )
            continue
        base_value = sorted(by_variant[baseline])[len(by_variant[baseline]) // 2]
        for variant in sorted(by_variant):
            if variant == baseline:
                continue
            value = sorted(by_variant[variant])[len(by_variant[variant]) // 2]
            if metric_names[key].get(variant) != metric_names[key].get(baseline):
                lines.append(
                    f"- `{dataset}` w{workers} b{batch} e{epochs} {device}, "
                    f"`{variant}`: different metric reported "
                    f"({metric_names[key].get(variant)} vs "
                    f"{metric_names[key].get(baseline)}), not comparable."
                )
                continue
            diff = relative_diff(value, base_value)
            if diff >= fail:
                problems += 1
                tag = "**FAIL**"
            elif diff >= warn:
                tag = "**WARN**"
            else:
                continue
            lines.append(
                f"- {tag} `{dataset}` w{workers} b{batch} e{epochs} {device}, "
                f"`{variant}`: {value:.6g} vs `{baseline}` {base_value:.6g} "
                f"({diff:.0%} relative difference)"
            )
    if problems == 0:
        lines.append("No cross-variant difference reached the FAIL threshold.")
    return lines, problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument(
        "--reproducibility-tolerance",
        type=float,
        default=1e-6,
        help="relative spread allowed between repeats of the same cell",
    )
    parser.add_argument("--warn-threshold", type=float, default=0.05)
    parser.add_argument("--fail-threshold", type=float, default=0.50)
    args = parser.parse_args()

    rows = load_rows(args.cells_csv)
    repro_lines, repro_problems = check_reproducibility(
        rows, args.reproducibility_tolerance
    )
    cross_lines, cross_problems = check_cross_variant(
        rows, args.baseline, args.warn_threshold, args.fail_threshold
    )

    header = ["# Correctness check", ""]
    body = header + repro_lines + cross_lines + [
        "",
        "This checks one scalar (`best_val_metric`) from one fixed seed over "
        "a few epochs on tiny data — it can catch a variant that's training "
        "on the wrong thing, but it is not a substitute for a real "
        "loss/gradient-level equivalence test.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(body) + "\n")

    if repro_problems or cross_problems:
        import sys

        sys.stderr.write(
            f"check_correctness: {repro_problems} reproducibility issue(s), "
            f"{cross_problems} cross-variant FAIL(s) — see {args.out}\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
