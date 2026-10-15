"""Exclusion lists of bad MAD-CORE structures for mlip.cpp `--exclude`.

Two kinds of flags, agreed on 2026-10-15:

- label flags (all splits, independent of any model): non-finite labels; a
  C-C distance below --cc A (physically absurd geometry); and a short contact
  (closest pair below --ratio of the covalent sum) whose largest label force
  is below --frac of the ZBL largest force in its header (where that exceeds
  10 eV/A), i.e. a label that misses the repulsion. (The header's ZBL energy
  is positive for 99.94 % of all structures, so it is no test.)
- fit flags (train only): per-structure force RMSE of a trained model on its
  own training data above --factor times the median of the structure's
  source group (a label the model cannot fit after seeing it) and above an
  absolute floor of --floor times the median over all training structures
  (so groups the model fits very well don't lose ordinary structures).

Writes OUT/rows.npz (the flagged MAD-CORE rows per flag), OUT/summary.txt,
and for every coreset size 2^K in --levels the directory OUT/2powK/ with
{train,val,test}.npy split indices, ready for `--exclude`.

    python make_exclusions.py OUT --labelcheck DIR --residuals PREFIX [--cc 0.8 --ratio 0.5 --frac 0.3 --factor 10 --floor 25]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # make_split, also under python -I
from make_split import load_split  # noqa: E402

RECORD = np.dtype([("id", "<i8"), ("step", "<i4"), ("n", "<i4"), ("de", "<f4"), ("fse", "<f4")])
DATA = Path("/work/cosmo/boittier/lyra/madcore-data")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--labelcheck", type=Path, default=DATA / "labelcheck")
    ap.add_argument("--residuals", required=True, help="PREFIX of --dump-errors on the train split (PREFIX.test.bin)")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--cc", type=float, default=0.8)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--factor", type=float, default=10.0)
    ap.add_argument("--floor", type=float, default=25.0, help="absolute floor, in units of the overall train median")
    ap.add_argument("--levels", type=int, nargs="+", default=[18, 20, 22, 24])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    split = {k: v[v < 2 ** 24] for k, v in load_split().items()}
    g = np.load(DATA / "groups.npz")
    names = g["groups"]
    flagged, lines = {}, []

    for s in ("train", "val", "test"):
        d = np.load(a.labelcheck / f"{s}.npz")
        er, ze, zf, mf, r, cc = d["e_rel"], d["zbl_e"], d["zbl_f"], d["max_force"], d["min_ratio"], d["min_cc"]
        nonfinite = ~np.isfinite(er) | ~np.isfinite(mf)
        absurd_cc = cc < a.cc
        no_wall = (r < a.ratio) & (zf > 10) & (mf < a.frac * zf)
        flagged[f"{s}_nonfinite"] = split[s][nonfinite]
        flagged[f"{s}_label"] = split[s][(absurd_cc | no_wall) & ~nonfinite]
        lines.append(f"{s}: {len(er):,} structures; C-C < {a.cc} A {absurd_cc.sum():,}; contact < {a.ratio} of the covalent "
                     f"sum with label force < {a.frac} x ZBL {no_wall.sum():,}; non-finite {nonfinite.sum():,}")

    # fit flags: the trained model's per-structure force RMSE on its training data
    res = np.fromfile(f"{a.residuals}.test.bin", RECORD)
    f_rmse = np.sqrt(res["fse"] / (3 * res["n"]))
    grp = g["group"][split["train"][res["id"]]]
    med = np.zeros(grp.max() + 1)
    for k in np.unique(grp):
        med[k] = np.median(f_rmse[grp == k])
    floor = a.floor * np.median(f_rmse)
    bad = (f_rmse > a.factor * med[grp]) & (f_rmse > floor)
    flagged["train_fit"] = split["train"][res["id"][bad]]
    lines.append(f"train fit flag: {bad.sum():,} of {len(res):,} scored structures have force RMSE > {a.factor:g}x "
                 f"their group's median and > {a.floor:g}x the overall median ({1e3 * floor:.0f} meV/A) "
                 f"(largest group medians: " +
                 ", ".join(f"{names[k]} {1e3 * med[k]:.0f}" for k in np.argsort(-np.bincount(grp))[:5]) + " meV/A)")

    remove = {s: np.union1d(np.union1d(flagged[f"{s}_nonfinite"], flagged[f"{s}_label"]),
                            flagged["train_fit"] if s == "train" else np.zeros(0, np.int64)) for s in ("train", "val", "test")}
    for s in remove:
        lines.append(f"{s}: remove {len(remove[s]):,} rows ({len(remove[s]) / len(split[s]):.3%})")
    # per group
    lines.append("\nper group (train): flagged by label / by fit / union, of the group's train structures")
    gt = g["group"][split["train"]]
    tot = np.bincount(gt, minlength=len(names))
    cl = np.bincount(g["group"][flagged["train_label"]], minlength=len(names))
    cf = np.bincount(g["group"][flagged["train_fit"]], minlength=len(names))
    cu = np.bincount(g["group"][remove["train"]], minlength=len(names))
    for k in np.argsort(-cu)[:25]:
        if cu[k]:
            lines.append(f"  {names[k]:36s} {cl[k]:>8,} {cf[k]:>8,} {cu[k]:>8,} of {tot[k]:>10,} ({cu[k] / tot[k]:.2%})")
    rng = np.random.default_rng(0)
    for key in ("train_label", "train_fit"):
        v = flagged[key]
        if len(v):
            lines.append(f"sample {key} rows: {np.sort(rng.choice(v, min(10, len(v)), replace=False)).tolist()}")

    np.savez(a.out / "rows.npz", **flagged, **{f"remove_{s}": v for s, v in remove.items()},
             params=np.array([a.ratio, a.cc, a.frac, a.factor, a.floor]))
    for lvl in a.levels:
        d = a.out / f"2pow{lvl}"
        d.mkdir(exist_ok=True)
        for s in ("train", "val", "test"):
            entries = split[s][split[s] < 2 ** lvl]
            rows = remove[s][remove[s] < 2 ** lvl]
            np.save(d / f"{s}.npy", np.searchsorted(entries, rows).astype(np.int64))
        lines.append(f"2^{lvl}: train {int((remove['train'] < 2 ** lvl).sum()):,}, val {int((remove['val'] < 2 ** lvl).sum()):,}, "
                     f"test {int((remove['test'] < 2 ** lvl).sum()):,} excluded")
    text = "\n".join(lines)
    (a.out / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
