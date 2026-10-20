"""A/B of an --exclude list: test errors of baseline vs exclusion runs (mean +- spread over seeds).

Both arms are scored on the same 2^24 test split; "clean" also drops the test
rows flagged in EXCLUDE/test.npy, "diverse" is the test rows below 2^16, and
the per-group table covers the groups the exclusion list touches most.

    python ab_exclude.py --a RUN [RUN ...] --b RUN [RUN ...] [--test-exclude DIR] [--train-exclude FILE]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # make_split, also under python -I
from controlled_curves import GROUPS, RECORD, rmse  # noqa: E402
from make_split import load_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", nargs="+", type=Path, required=True, help="baseline run dirs")
    ap.add_argument("--b", nargs="+", type=Path, required=True, help="runs with --exclude")
    ap.add_argument("--test-exclude", type=Path, default=Path("/work/cosmo/boittier/lyra/madcore-data/exclude/2pow24/test.npy"))
    ap.add_argument("--train-exclude", type=Path,
                    default=Path("/work/cosmo/boittier/lyra/madcore-data/exclude/2pow18-trainonly/train.npy"))
    a = ap.parse_args()
    split = load_split()
    test_rows = split["test"][split["test"] < 2 ** 24]
    g = np.load(GROUPS)
    names, tgrp = g["groups"], g["group"][test_rows]
    flagged = np.zeros(len(test_rows), bool)
    flagged[np.load(a.test_exclude)] = True
    # groups that lose most training rows (rows of the run's subset, = 2^K train split for an FPS prefix)
    sub = np.load(a.b[0] / "rows.npy")
    lost = g["group"][sub[np.load(a.train_exclude)]]
    top = np.argsort(-np.bincount(lost, minlength=len(names)))[:6]

    sets = {"uniform": lambda r: r, "clean": lambda r: r[~flagged[r["id"]]],
            "diverse": lambda r: r[test_rows[r["id"]] < 2 ** 16]}
    sets |= {str(names[k]): (lambda k: lambda r: r[tgrp[r["id"]] == k])(k) for k in top}
    arms = {}
    for arm, dirs in (("A base", a.a), ("B excl", a.b)):
        recs = [np.fromfile(d / "errors.test.bin", RECORD) for d in dirs]
        arms[arm] = {s: np.array([rmse(f(r)) for r in recs], float) for s, f in sets.items()}
    print(f"train rows excluded: {len(lost):,} (top groups: " +
          ", ".join(f"{names[k]} {np.sum(lost == k)}" for k in top) + f"); test rows flagged: {flagged.sum()}")
    print(f"{'test set':28s} {'n':>7s} " + " ".join(f"{arm + ' E':>15s} {arm + ' F':>16s}" for arm in arms) + f" {'dF':>7s}")
    r0 = np.fromfile(a.a[0] / "errors.test.bin", RECORD)
    for s, f in sets.items():
        cells, means = [], []
        for arm in arms:
            v = arms[arm][s]
            cells.append(f"{v[:, 0].mean():8.2f} ±{v[:, 0].std():5.2f} {v[:, 1].mean():9.1f} ±{v[:, 1].std():5.1f}")
            means.append(v[:, 1].mean())
        print(f"{s[:28]:28s} {len(f(r0)):7,d} " + " ".join(cells) + f" {means[1] - means[0]:+7.1f}")
    print(f"(meV/atom, meV/A; ± = std over {len(a.a)} / {len(a.b)} seeds)")


if __name__ == "__main__":
    main()
