"""Per-structure geometry checks of a MAD-CORE memmap split (bad structures).

For every structure: the closest contact relative to the covalent radii,
r = min_ij d_ij / (R_i + R_j) (ASE covalent radii), with the element pair
and distance where it occurs; the number of pairs below 0.5 / 0.6 / 0.7 of
the covalent sum; the shortest C-C distance; the largest covalent
coordination (neighbours within 1.25 (R_i + R_j)); the number density of 3D
periodic cells; the largest force norm and the energy per atom. Contacts are
searched within 0.75 of the largest covalent sum the structure can have, so
r >= 0.75 means "no short contact" (stored as 0.75). Entries are aligned
with the memmap split, so row = load_split()[split][i].

    python geometry_scan.py MEMMAP_DIR OUT_DIR [--splits train val test] [--workers 72]

writes OUT_DIR/{split}.npz.
"""

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import vesin
from ase.data import covalent_radii

RCOV = covalent_radii.astype(np.float64)
CAP = 0.75  # contacts are searched up to this fraction of the covalent sum
FIELDS = {
    "min_ratio": np.float32, "min_dist": np.float32, "min_z1": np.int16, "min_z2": np.int16,
    "n_below_05": np.int32, "n_below_06": np.int32, "n_below_07": np.int32,
    "min_cc": np.float32, "max_coord": np.int32, "density": np.float32,
    "max_force": np.float32, "energy_per_atom": np.float32, "n_atoms": np.int32,
}


class Split:
    def __init__(self, path):
        self.na = np.load(path / "na.npy")
        self.x = np.memmap(path / "x.bin", np.float32, "r").reshape(-1, 3)
        self.a = np.memmap(path / "a.bin", np.int32, "r")
        self.c = np.memmap(path / "c.bin", np.float32, "r").reshape(-1, 3, 3)
        self.e = np.memmap(path / "energy.bin", np.float32, "r")
        self.f = np.memmap(path / "non_conservative_force.bin", np.float32, "r").reshape(-1, 3)

    def __len__(self):
        return len(self.na) - 1


def box_of(cell):
    """vesin box and periodicity; partial periodicity (slabs, wires) gets long
    vacuum vectors orthogonal to the periodic rows, as mlip.cpp does."""
    pbc = np.any(cell != 0, axis=1)
    if not pbc.any():
        return np.zeros((3, 3)), False
    box = cell.astype(np.float64).copy()
    rows, empty = box[pbc], np.flatnonzero(~pbc)
    if len(rows) == 2:  # slab
        n = np.cross(rows[0], rows[1])
        box[empty[0]] = 1000.0 * n / np.linalg.norm(n)
    elif len(rows) == 1:  # wire
        p = rows[0]
        u = np.cross(p, np.eye(3)[np.argmin(np.abs(p))])
        v = np.cross(p, u)
        box[empty[0]], box[empty[1]] = 1000.0 * u / np.linalg.norm(u), 1000.0 * v / np.linalg.norm(v)
    return box, True


def scan_range(args):
    path, lo, hi = args
    s = Split(Path(path))
    out = {k: np.zeros(hi - lo, t) for k, t in FIELDS.items()}
    nl = vesin.NeighborList(cutoff=1.0, full_list=False)
    for i in range(lo, hi):
        a0, a1 = s.na[i], s.na[i + 1]
        z = np.asarray(s.a[a0:a1])
        x = np.asarray(s.x[a0:a1], np.float64)
        f = np.asarray(s.f[a0:a1])
        j = i - lo
        out["n_atoms"][j] = len(z)
        out["energy_per_atom"][j] = s.e[i] / max(len(z), 1)
        out["max_force"][j] = np.sqrt((f * f).sum(1)).max() if len(z) else 0
        out["min_ratio"][j], out["min_dist"][j], out["min_cc"][j] = CAP, np.inf, np.inf
        cell = np.asarray(s.c[i])
        pbc = np.any(cell != 0, axis=1)
        if pbc.all():
            vol = abs(np.linalg.det(cell.astype(np.float64)))
            out["density"][j] = len(z) / vol if vol > 0 else np.inf
        if len(z) < 2 and not pbc.any():
            continue
        r = RCOV[z]
        nl.cutoff = 1.25 * 2 * r.max()  # also the coordination shell
        box, periodic = box_of(cell)
        ii, jj, d = nl.compute(x, box, periodic, quantities="ijd")
        if not len(d):
            continue
        ratio = d / (r[ii] + r[jj])
        coord = np.bincount(np.concatenate([ii[ratio < 1.25], jj[ratio < 1.25]]), minlength=len(z))
        out["max_coord"][j] = coord.max()
        k = np.argmin(ratio)
        if ratio[k] < CAP:
            out["min_ratio"][j], out["min_dist"][j] = ratio[k], d[k]
            out["min_z1"][j], out["min_z2"][j] = sorted((z[ii[k]], z[jj[k]]))
        out["n_below_05"][j] = (ratio < 0.5).sum()
        out["n_below_06"][j] = (ratio < 0.6).sum()
        out["n_below_07"][j] = (ratio < 0.7).sum()
        cc = (z[ii] == 6) & (z[jj] == 6)
        if cc.any():
            out["min_cc"][j] = d[cc].min()
    return lo, out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("memmap", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--limit", type=int, default=0, help="first N structures per split (testing)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.splits:
        t = time.time()
        n = len(Split(a.memmap / name))
        n = min(n, a.limit) if a.limit else n
        jobs = [(str(a.memmap / name), lo, min(n, lo + a.chunk)) for lo in range(0, n, a.chunk)]
        res = {k: np.zeros(n, t_) for k, t_ in FIELDS.items()}
        with Pool(a.workers) as pool:
            for lo, out in pool.imap_unordered(scan_range, jobs):
                for k in res:
                    res[k][lo:lo + len(out[k])] = out[k]
        tmp = a.out / f".{name}.npz.tmp.npz"
        try:
            np.savez(tmp, **res)
            tmp.rename(a.out / f"{name}.npz")
        finally:
            tmp.unlink(missing_ok=True)
        r = res["min_ratio"]
        print(f"{name}: {n:,} structures in {time.time() - t:.0f} s; min_ratio < 0.5: {(r < 0.5).sum():,}, "
              f"< 0.6: {(r < 0.6).sum():,}, < 0.7: {(r < 0.7).sum():,}; "
              f"C-C < 1.0 A: {(res['min_cc'] < 1.0).sum():,}", flush=True)


if __name__ == "__main__":
    main()
