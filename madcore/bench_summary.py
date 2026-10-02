"""Summarize a bench-pipeline.sbatch job: one row per cell.

    python bench_summary.py <jobid>
"""

import datetime as dt
import re
import statistics
import sys
from pathlib import Path

ROOT = Path("/work/cosmo/boittier/kuma/madcore-lorem")
job = sys.argv[1]
log = (ROOT / "logs").glob(f"lorem-bench-pipe-{job}.out")
walls = dict(re.findall(r"=== CELL \S+/(\S+) exit \d+ wall ([0-9.]+) s", next(log).read_text()))
stamp = lambda line: dt.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S")

print(f"{'cell':34s} {'setup s':>7s} {'epoch s':>7s} {'epochs':>6s} {'wall s':>7s}  epoch-0 train loss / val E RMSE")
for cell in sorted((ROOT / "bench" / job).iterdir()):
    text = (cell / "train.out").read_text() if (cell / "train.out").exists() else ""
    lines = text.splitlines()
    executed = next((stamp(l) for l in lines if "Executed command" in l), None)
    started = next((stamp(l) for l in lines if l.endswith("- Starting training")), None)
    epochs = [(stamp(l), l) for l in lines if "- Epoch:" in l]
    times = [(b[0] - a[0]).total_seconds() for a, b in zip(epochs, epochs[1:])]
    first = re.search(r"training loss: (\S+) .*validation energy RMSE \(per atom\): +([0-9.]+)", epochs[0][1]) if epochs else None
    print(
        f"{cell.name:34s} {(started - executed).total_seconds() if started and executed else float('nan'):7.0f} "
        f"{statistics.median(times) if times else float('nan'):7.1f} {len(epochs):6d} "
        f"{float(walls.get(cell.name, 'nan')):7.0f}  {first.group(1) + ' / ' + first.group(2) if first else '-'}"
    )
    if "training-step timing" in text:
        report = text[text.rindex("training-step timing") :].split("\n[20")[0]
        print("    " + "\n    ".join(report.splitlines()[1:]))
