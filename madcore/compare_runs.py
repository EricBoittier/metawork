"""Compare long-range vs short-range LOREM runs epoch by epoch, from train.csv.

For each subset, prints the latest common epoch of the -lr and -sr runs and
the long-range change in validation energy/force RMSE (negative = LR better),
plus a few earlier checkpoints.

    python compare_runs.py [--suffix 100ep] [--every 10] [--brief]
"""

import argparse
import csv
import glob

ROOT = "/work/cosmo/boittier/kuma/madcore-lorem/runs"
COLUMNS = {
    "lr": "learning rate",
    "train_loss": "training loss",
    "train_E": "training energy RMSE (per atom)",
    "train_F": "training forces RMSE",
    "val_loss": "validation loss",
    "val_E": "validation energy RMSE (per atom)",
    "val_F": "validation forces RMSE",
}

p = argparse.ArgumentParser()
p.add_argument("--suffix", default="100ep")
p.add_argument("--every", type=int, default=10)
p.add_argument("--brief", action="store_true", help="one line per subset")
args = p.parse_args()


def history(run):
    """Epoch -> metrics, over every output directory of the run (segments)."""
    rows = {}
    for path in sorted(glob.glob(f"{ROOT}/{run}/outputs/*/*/train.csv")):
        with open(path) as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader)]
            next(reader)  # units
            for line in reader:
                if not line or not line[0].strip():
                    continue
                values = dict(zip(header, (v.strip() for v in line)))
                rows[int(values["Epoch"])] = {k: float(values[c]) for k, c in COLUMNS.items()}
    return rows


def change(lr, sr):
    return 100 * (lr - sr) / sr


for subset in ("mixed", "periodic", "nonperiodic"):
    lr_run, sr_run = (f"lorem-{subset}-{tag}-{args.suffix}" for tag in ("lr", "sr"))
    lr, sr = history(lr_run), history(sr_run)
    common = sorted(set(lr) & set(sr))
    if not common:
        print(f"{subset:12s} no common epoch yet (lr {len(lr)}, sr {len(sr)} epochs)")
        continue
    last = common[-1]
    a, b = lr[last], sr[last]
    summary = (
        f"{subset:12s} epoch {last:3d} (lr {max(lr)}, sr {max(sr)} done) | "
        f"val E {a['val_E']:7.1f} vs {b['val_E']:7.1f} ({change(a['val_E'], b['val_E']):+5.1f}%) | "
        f"val F {a['val_F']:7.1f} vs {b['val_F']:7.1f} ({change(a['val_F'], b['val_F']):+5.1f}%)"
    )
    print(summary)
    if args.brief:
        continue
    print(f"  {'epoch':>5s} {'lr':>9s} | {'val E LR':>8s} {'val E SR':>8s} {'dE%':>6s} | "
          f"{'val F LR':>8s} {'val F SR':>8s} {'dF%':>6s} | {'train E LR':>10s} {'train E SR':>10s}")
    shown = [e for e in common if e % args.every == 0 or e == last]
    for e in shown:
        a, b = lr[e], sr[e]
        print(f"  {e:5d} {a['lr']:9.2e} | {a['val_E']:8.1f} {b['val_E']:8.1f} "
              f"{change(a['val_E'], b['val_E']):+6.1f} | {a['val_F']:8.1f} {b['val_F']:8.1f} "
              f"{change(a['val_F'], b['val_F']):+6.1f} | {a['train_E']:10.1f} {b['train_E']:10.1f}")
