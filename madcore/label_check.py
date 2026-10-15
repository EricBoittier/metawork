"""Physics check of MAD-CORE labels at short contacts (bad structures, part 1).

Per structure of each 2^24 split: the teacher energy above a linear
per-element baseline (least squares on a sample of the train split), per
atom; the ZBL repulsion of the structure from its header (energy per atom,
largest force); the label's largest force; the geometry scan's closest
contact ratio and shortest C-C. A short contact whose label is far below the
ZBL repulsion is a broken label rather than a deliberate high-energy
sample. Writes OUT/{split}.npz with these per-entry arrays and prints the
joint distributions needed to choose the cut.

    python label_check.py OUT [--sample 2000000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # make_split, also under python -I
from make_split import load_split  # noqa: E402

MM = Path("/home/boittier/data/madcore/memmap-2pow24")
DATA = Path("/work/cosmo/boittier/lyra/madcore-data")


def composition_fit(split, n_sample, rng):
    na = np.load(split / "na.npy")
    a = np.memmap(split / "a.bin", np.int32, "r")
    e = np.memmap(split / "energy.bin", np.float32, "r")
    pick = np.sort(rng.choice(len(na) - 1, min(n_sample, len(na) - 1), replace=False))
    X = np.zeros((len(pick), 119), np.float64)
    for k, i in enumerate(pick):
        X[k] += np.bincount(a[na[i]:na[i + 1]], minlength=119)
    y = np.asarray(e[pick], np.float64)
    used = X.any(0)
    coef = np.zeros(119)
    coef[used] = np.linalg.lstsq(X[:, used], y, rcond=None)[0]
    return coef


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--sample", type=int, default=2_000_000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    coef = composition_fit(MM / "train", a.sample, rng)
    np.save(a.out / "composition.npy", coef)
    g = np.load(DATA / "groups.npz")
    zbl_e, zbl_f = g["zbl_energy_per_atom"], g["zbl_max_force"]
    split_rows = load_split()
    for name in ("train", "val", "test"):
        d = MM / name
        na = np.load(d / "na.npy")
        z = np.asarray(np.memmap(d / "a.bin", np.int32, "r"))
        e = np.asarray(np.memmap(d / "energy.bin", np.float32, "r"), np.float64)
        n = np.diff(na)
        base = np.add.reduceat(coef[z], na[:-1]) if len(z) else np.zeros(0)
        rows = split_rows[name]
        rows = rows[rows < 2 ** 24]
        geo = np.load(DATA / "geometry" / f"{name}.npz")
        out = {"e_rel": ((e - base) / n).astype(np.float32), "zbl_e": zbl_e[rows], "zbl_f": zbl_f[rows],
               "max_force": geo["max_force"], "min_ratio": geo["min_ratio"], "min_cc": geo["min_cc"],
               "group": g["group"][rows]}
        tmp = a.out / f".{name}.tmp.npz"
        try:
            np.savez(tmp, **out)
            tmp.rename(a.out / f"{name}.npz")
        finally:
            tmp.unlink(missing_ok=True)
        if name != "train":
            continue
        q = [0.1, 0.5, 0.9, 0.99]
        er, ze, zf, mf, r, cc = out["e_rel"], out["zbl_e"], out["zbl_f"], out["max_force"], out["min_ratio"], out["min_cc"]
        print(f"train: {len(er):,} structures; energy above composition baseline (eV/atom) quantiles {np.quantile(er, q).round(2)}")
        print(f"ZBL energy per atom > 0 in {np.mean(ze > 0):.2%}; quantiles of nonzero {np.quantile(ze[ze > 0], q).round(3)}")
        for lab, m in (("min ratio < 0.5", r < 0.5), ("0.5 <= ratio < 0.7", (r >= 0.5) & (r < 0.7)),
                       ("ratio >= 0.7", r >= 0.7), ("C-C < 1.0 A", cc < 1.0)):
            print(f"\n{lab}: {m.sum():,}")
            print(f"  e_rel q {np.quantile(er[m], q).round(2)}   zbl_e q {np.quantile(ze[m], q).round(2)}")
            print(f"  label max F q {np.quantile(mf[m], q).round(1)}   zbl max F q {np.quantile(zf[m], q).round(1)}")
            for gap in (0, 2, 5, 10):
                print(f"  e_rel < zbl_e - {gap:>2} eV/atom: {(m & (er < ze - gap)).sum():>9,}", end="")
            print()
            for frac in (0.01, 0.1, 0.3):
                print(f"  label max F < {frac} x ZBL max F (ZBL > 10 eV/A): {(m & (zf > 10) & (mf < frac * zf)).sum():>9,}", end="")
            print()


if __name__ == "__main__":
    main()
