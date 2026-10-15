"""A training subset of the MAD-CORE 2^24 memmap, for controlled learning curves.

Picks N structures of the 2^24 train split: the first N in farthest-point
(FPS) order (`--kind fps`, = the 2^K coreset's train split when N is its
size) or N drawn uniformly at random (`--kind random`), optionally only
those of one dataset_group (`--group OMat24`). Writes a MemmapDataset with
that train/ split and val/, test/ linked to the given splits (so every
subset is scored on the same test set), plus rows.npy: the MAD-CORE rows
of the chosen training structures.

    python make_subset_memmap.py --src memmap-2pow24 --kind random --n 1027000 --val memmap-2pow20/val --test memmap-2pow24/test -o OUT
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # make_split, also under python -I
from make_split import load_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="the 2^24 memmap (train/ val/ test/)")
    ap.add_argument("--kind", choices=["fps", "random"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--group", help="only this dataset_group")
    ap.add_argument("--groups", type=Path, default=Path("/work/cosmo/boittier/lyra/madcore-data/groups.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()

    src = a.src / "train"
    na = np.load(src / "na.npy")
    rows = load_split()["train"]
    rows = rows[rows < 2 ** 24]
    assert len(rows) == len(na) - 1, "split and memmap disagree"
    pool = np.arange(len(rows))
    if a.group:
        g = np.load(a.groups)
        code = int(np.flatnonzero(g["groups"] == a.group)[0])
        pool = pool[g["group"][rows] == code]
    if a.n > len(pool):
        sys.exit(f"--n {a.n} > {len(pool)} available")
    pick = pool[:a.n] if a.kind == "fps" else np.sort(np.random.default_rng(a.seed).choice(pool, a.n, replace=False))

    counts = na[pick + 1] - na[pick]
    new_na = np.concatenate([[0], np.cumsum(counts)])
    starts = na[pick]
    # atom index of every chosen atom, in order
    atoms = np.repeat(starts - new_na[:-1], counts) + np.arange(new_na[-1])

    dst = a.out / "train"
    dst.mkdir(parents=True, exist_ok=True)
    np.save(dst / "na.npy", new_na.astype(np.int64))
    np.save(dst / "ns.npy", np.array(len(pick), dtype=np.int64))
    np.save(a.out / "rows.npy", rows[pick])
    per_atom = {"x.bin": (np.float32, 3), "a.bin": (np.int32, 1), "non_conservative_force.bin": (np.float32, 3)}
    per_system = {"c.bin": (np.float32, 9), "energy.bin": (np.float32, 1)}
    for name, (dt, w) in per_atom.items():
        m = np.memmap(src / name, dt, "r").reshape(-1, w)
        m[atoms].tofile(dst / name)
    for name, (dt, w) in per_system.items():
        m = np.memmap(src / name, dt, "r").reshape(-1, w)
        m[pick].tofile(dst / name)
    for split, target in (("val", a.val), ("test", a.test)):
        link = a.out / split
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.resolve())
    print(f"{a.out}: {len(pick):,} structures ({a.kind}{', ' + a.group if a.group else ''}), "
          f"{new_na[-1]:,} atoms, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
