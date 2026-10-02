"""Submit the six LOREM runs, sized from the canaries, under a CHF ceiling.

Runs after both canaries (afterok). Reads ``ELAPSED`` from the canary logs,
turns it into seconds per structure (train + val + test of the 8k subset,
setup included, so it overestimates), and projects each full run:

    T = margin * rate * (epochs * (n_train + n_val) + n_test) + overhead

Each run becomes ceil(T / segment) segments of ``--segment-hours`` chained
with afterany (train-lorem-pbc.sbatch resumes and stops on a crash). If the
summed bound (segments x wall x CHF/GPU-h) exceeds ``--ceiling``, nothing is
submitted. Everything is appended to RUNS.md.
"""

import argparse
import datetime
import io
import math
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = Path("/work/cosmo/boittier/kuma/madcore-lorem")
CHF_PER_GPU_H = 0.52  # kuma h100, clusters/kuma.md


def elapsed(job_name):
    logs = sorted((ROOT / "logs").glob(f"{job_name}-*.out"))
    found = re.findall(r"^ELAPSED (\d+)$", logs[-1].read_text(), re.M)
    return int(found[-1])


def n_structures(subset, split):
    with zipfile.ZipFile(ROOT / "data" / subset / f"{split}.zip") as z:
        return len(np.load(io.BytesIO(z.read("metadata/atom_counts.npy"))))


def sbatch(args):
    out = subprocess.run(["sbatch", "--parsable", *args], capture_output=True, text=True, check=True)
    return out.stdout.strip().split(";")[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ceiling", type=float, default=100.0, help="CHF, summed bound of all segments")
    parser.add_argument("--segment-hours", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--margin", type=float, default=1.3)
    parser.add_argument("--overhead-hours", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    canary = {"true": "canary-mixed8k-lr", "false": "canary-mixed8k-sr"}
    canary_n = sum(n_structures("mixed-8192", s) for s in ("train", "val", "test"))
    rate = {lr: elapsed(name) / canary_n for lr, name in canary.items()}

    plan = []
    for subset in ("mixed", "periodic", "nonperiodic"):
        n_train, n_val, n_test = (n_structures(subset, s) for s in ("train", "val", "test"))
        for lr, tag in (("true", "lr"), ("false", "sr")):
            hours = args.margin * rate[lr] * (args.epochs * (n_train + n_val) + n_test) / 3600 + args.overhead_hours
            segments = math.ceil(hours / args.segment_hours)
            plan.append((f"lorem-{subset}-{tag}", subset, lr, hours, segments))

    bound = sum(s for *_, s in plan) * args.segment_hours * CHF_PER_GPU_H
    lines = [f"\n## dispatch {datetime.datetime.now():%Y-%m-%d %H:%M}\n",
             f"canary s/structure: lr {rate['true']:.4f}, sr {rate['false']:.4f} (n={canary_n})  ",
             f"projected (x{args.margin} + {args.overhead_hours} h), {args.segment_hours} h segments, bound CHF {bound:.2f}, ceiling CHF {args.ceiling:.2f}\n"]
    lines += [f"- {name}: {hours:.1f} h -> {segments} segment(s)" for name, _, _, hours, segments in plan]

    if bound > args.ceiling or args.dry_run:
        lines.append(f"\n**not submitted** ({'dry run' if args.dry_run else 'bound above ceiling'}); ask Eric.")
    else:
        for name, subset, lr, _, segments in plan:
            prev = None
            for k in range(segments):
                export = f"ALL,SUBSET={subset},LONG_RANGE={lr}" + (f",PREV_JOB={prev}" if prev else "")
                extra = [f"--dependency=afterany:{prev}"] if prev else []
                prev = sbatch(["-J", name, f"--time={args.segment_hours}:00:00", f"--export={export}", *extra,
                               str(HERE / "train-lorem-pbc.sbatch")])
                lines.append(f"  - submitted {name} segment {k + 1}/{segments}: job {prev}")
    report = "\n".join(lines) + "\n"
    print(report)
    with open(HERE / "RUNS.md", "a") as f:
        f.write(report)


if __name__ == "__main__":
    main()
