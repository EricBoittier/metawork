"""Check that variants train to the same place, not just at the same speed.

Every other check in this harness is about wall-clock speed. This is the
one check that looks at whether the six variants are actually computing the
same thing. It relies on each branch's `benchmarks/benchmark_pipeline.py`:

- seeding `random`/`numpy`/`torch` with a fixed `--seed` before building the
  dataset or model, so model init, augmentation, and data shuffling draw the
  same sequence of random numbers across variants for a given seed
- reporting `trainer.best_metric` (the best validation metric seen across
  training, tracked by `Trainer.train` regardless of `log_interval`) as a
  `best_val_metric ... (metric_name) at epoch ...` line
- reporting the *training* and *validation* loss after epoch 1 specifically
  (`epoch1_metrics train_loss=... val_loss=...`), read straight off the raw
  metrics dict `Trainer.train` builds internally (via a `MetricLogger.log`
  interception, not text-parsing a log line) — closer to a real
  loss-equivalence check than `best_val_metric` alone, since it's a single
  well-defined point (one epoch in) rather than whichever epoch happened to
  score best over a run's whole, noisier trajectory

Three things get checked, all from `results/cells.csv`:

1. **Reproducibility**: repeats of the *same* (variant, dataset, workers,
   batch, epochs, seed, device) should agree on all three metrics — same
   seed, same computation. A mismatch means something is nondeterministic
   that the seeding assumption didn't actually cover (e.g. a worker-level
   RNG not seeded from the base seed), and every cross-variant comparison
   for that cell is suspect.
2. **Cross-variant equivalence**: within a (dataset, workers, batch, epochs,
   seed, device) group, every variant's metrics are compared against
   `baseline`'s. A real difference here means two variants are not
   computing the same thing — the exact failure mode no throughput number
   can catch. Done per seed, not pooled, so a variant can't look consistent
   by accident of averaging over seeds that disagree in different
   directions.
3. **Seed stability**: for each (variant, dataset, workers, batch, epochs,
   device), how much does that variant's `best_val_metric` disagreement
   with baseline vary *across* the tested seeds? A gap that's ~stable
   across seeds is more likely a systematic code difference; one that
   swings widely is more likely just noise this test's few epochs on tiny
   data can't average out.

This is still not a full correctness proof — three scalars, a handful of
seeds, a few epochs on tiny data. Treat FAIL as "investigate," and OK as "no
problem detected at this resolution," not as "proven correct."
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


METRICS = ["best_val_metric", "epoch1_train_loss", "epoch1_val_loss"]

ReproKey = Tuple[str, str, str, str, str, str, str]  # +metric
CrossKey = Tuple[str, str, str, str, str, str]  # dataset..seed, no variant


def load_rows(cells_csv) -> List[Dict[str, str]]:
    import csv

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
    groups: Dict[ReproKey, List[float]] = defaultdict(list)
    for row in rows:
        for metric_name in METRICS:
            value = to_float(row.get(metric_name, ""))
            if value is None:
                continue
            key = (
                metric_name,
                row["variant"],
                row["dataset"],
                row["num_workers"],
                row["batch_size"],
                row["epochs"],
                row.get("seed", "0"),
            )
            groups[key].append(value)

    lines = [
        "## Reproducibility (same variant, config, and seed, different repeats)",
        "",
    ]
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
            metric_name, variant, dataset, workers, batch, epochs, seed = key
            lines.append(
                f"- **FAIL** `{metric_name}` for `{variant}`/`{dataset}` "
                f"(w{workers} b{batch} e{epochs} seed{seed}): repeats disagree "
                f"— {sorted(values)}. The fixed seed didn't make this cell "
                "deterministic; treat cross-variant comparisons for this "
                "cell as unreliable until that's understood."
            )
    if tested == 0:
        lines.append(
            "No (metric, variant, config, seed) cell had >=2 repeats to "
            "compare — nothing was tested here."
        )
    elif problems == 0:
        lines.append(
            f"All {tested} repeated cell(s) reproduced themselves within "
            f"{tolerance:.0%}, across all of {', '.join(METRICS)}."
        )
    return lines, problems


def _median(values: List[float]) -> float:
    return sorted(values)[len(values) // 2]


def check_cross_variant(
    rows: List[Dict[str, str]], baseline: str, warn: float, fail: float
) -> Tuple[List[str], int, Dict[Tuple[str, CrossKey], float]]:
    # groups[metric][cross_key][variant] -> list of values (over repeats)
    groups: Dict[str, Dict[CrossKey, Dict[str, List[float]]]] = {
        m: defaultdict(lambda: defaultdict(list)) for m in METRICS
    }
    for row in rows:
        key: CrossKey = (
            row["dataset"],
            row["num_workers"],
            row["batch_size"],
            row["epochs"],
            row.get("seed", "0"),
            row["device"],
        )
        for metric_name in METRICS:
            value = to_float(row.get(metric_name, ""))
            if value is not None:
                groups[metric_name][key][row["variant"]].append(value)

    lines = [
        "",
        "## Cross-variant equivalence (same config and seed, different variants)",
        "",
        f"vs `{baseline}`, same (dataset, workers, batch, epochs, seed, device). "
        f"WARN >= {warn:.0%} relative difference, FAIL >= {fail:.0%}.",
        "",
    ]
    problems = 0
    # stashed for the seed-stability rollup: (metric, dataset, workers, batch,
    # epochs, device, variant) -> {seed: relative_diff}
    diffs_by_seed: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(dict)

    for metric_name in METRICS:
        by_key = groups[metric_name]
        for key in sorted(by_key):
            dataset, workers, batch, epochs, seed, device = key
            by_variant = by_key[key]
            if baseline not in by_variant:
                continue
            base_value = _median(by_variant[baseline])
            for variant in sorted(by_variant):
                if variant == baseline:
                    continue
                value = _median(by_variant[variant])
                diff = relative_diff(value, base_value)
                rollup_key = (
                    metric_name,
                    dataset,
                    workers,
                    batch,
                    epochs,
                    device,
                    variant,
                )
                diffs_by_seed[rollup_key][seed] = diff
                if diff >= fail:
                    problems += 1
                    tag = "**FAIL**"
                elif diff >= warn:
                    tag = "**WARN**"
                else:
                    continue
                lines.append(
                    f"- {tag} `{metric_name}` `{dataset}` w{workers} b{batch} "
                    f"e{epochs} seed{seed} {device}, `{variant}`: "
                    f"{value:.6g} vs `{baseline}` {base_value:.6g} "
                    f"({diff:.0%} relative difference)"
                )
    if problems == 0:
        lines.append("No cross-variant difference reached the FAIL threshold.")
    return lines, problems, diffs_by_seed


def seed_stability(
    diffs_by_seed: Dict[Tuple[str, ...], Dict[str, float]], min_seeds: int
) -> List[str]:
    lines = [
        "",
        "## Seed stability (best_val_metric only, across tested seeds)",
        "",
        f"For (variant, config) pairs tested with >={min_seeds} seeds: how much "
        "does the vs-baseline relative difference move as the seed changes? "
        "A tight range says a WARN/FAIL above is probably systematic, not luck "
        "of one seed; a wide range says the opposite.",
        "",
    ]
    rows_out = []
    for key, by_seed in sorted(diffs_by_seed.items()):
        metric_name = key[0]
        if metric_name != "best_val_metric" or len(by_seed) < min_seeds:
            continue
        _, dataset, workers, batch, epochs, device, variant = key
        values = list(by_seed.values())
        rows_out.append(
            f"- `{dataset}` w{workers} b{batch} e{epochs} {device}, `{variant}`: "
            f"{min(values):.0%}-{max(values):.0%} across {len(values)} seeds "
            f"(mean {statistics.mean(values):.0%})"
        )
    if not rows_out:
        lines.append(
            f"No (variant, config) pair was tested with >={min_seeds} seeds — "
            "run with `--config correctness=true` for multi-seed coverage."
        )
    else:
        lines.extend(rows_out)
    return lines


def main() -> None:
    import argparse
    from pathlib import Path

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
    parser.add_argument(
        "--min-seeds-for-stability",
        type=int,
        default=2,
        help="minimum seeds tested before the stability rollup reports a pair",
    )
    args = parser.parse_args()

    rows = load_rows(args.cells_csv)
    repro_lines, repro_problems = check_reproducibility(
        rows, args.reproducibility_tolerance
    )
    cross_lines, cross_problems, diffs_by_seed = check_cross_variant(
        rows, args.baseline, args.warn_threshold, args.fail_threshold
    )
    stability_lines = seed_stability(diffs_by_seed, args.min_seeds_for_stability)

    header = ["# Correctness check", ""]
    body = header + repro_lines + cross_lines + stability_lines + [
        "",
        f"This checks {len(METRICS)} scalars ({', '.join(METRICS)}) from a "
        "handful of fixed seeds over a few epochs on tiny data — it can catch "
        "a variant that's training on the wrong thing or isn't as "
        "seed-independent as assumed, but it is not a substitute for a real "
        "gradient-level equivalence test.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(body) + "\n")

    if repro_problems or cross_problems:
        # Deliberately exit 0: Snakemake deletes a rule's output the moment
        # its shell command exits non-zero, and the whole point of this
        # report is to survive to be read *especially* when it found a
        # problem. Read the report, or grep it for FAIL, rather than
        # gating a build on this script's exit code.
        sys.stderr.write(
            f"check_correctness: {repro_problems} reproducibility issue(s), "
            f"{cross_problems} cross-variant FAIL(s) — see {args.out}\n"
        )


if __name__ == "__main__":
    main()
