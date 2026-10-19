#!/usr/bin/env python3
"""Why mlip.cpp and metatrain disagree on a few structures: adaptive radii,
kept edges and pair margins, metatrain float64 / float32 vs mlip.cpp.

    cutoff_diag.py --ckpt model.ckpt --split SPLIT --edges-dir DIR IDX [IDX ...]

DIR/IDX.txt is `mlipcpp-md edges MODEL --data SPLIT --index IDX` output
(per atom: kept edges and the radius GraphModel uses). For each structure:
per-atom radius differences, edges whose keep decision differs (margin
d - (r_i + r_j)/2 on each side), and how much the float64 energy moves under
a 1e-6 A displacement (a jump far above F.dx means a keep decision flipped).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/boittier/metawork/mlip.cpp/scripts")
from roundtrip_compare import memmap  # noqa: E402

ATOM = re.compile(r"atom (\d+) Z (\d+): engine (\d+) cpu (\d+) kept (\d+) radius ([0-9.eE+-]+)")


def mlipcpp_atoms(path):
    rows = [ATOM.search(l) for l in Path(path).read_text().splitlines()]
    rows = [m for m in rows if m]
    E = re.search(r"E engine ([-0-9.]+)\s+cpu ([-0-9.]+)", Path(path).read_text())
    return (np.array([int(m[5]) for m in rows]), np.array([float(m[6]) for m in rows]),
            np.array([int(m[3]) for m in rows]), float(E[2]) if E else np.nan)


def evaluate(model, data, s, dtype, shift=None):
    from metatomic.torch import ModelOutput, System
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

    model = model.to(dtype).eval()
    lo, hi = data["na"][s], data["na"][s + 1]
    x = torch.tensor(data["x"][lo:hi], dtype=torch.float64)
    if shift is not None:
        x = x + shift
    cell = torch.tensor(data["c"][s], dtype=dtype)
    sys_ = System(types=torch.tensor(data["a"][lo:hi]), positions=x.to(dtype), cell=cell, pbc=(cell != 0).any(dim=1))
    sys_ = get_system_with_neighbor_lists(sys_, model.requested_neighbor_lists())
    out = model([sys_], {"energy": ModelOutput(), "mtt::aux::cutoff_stats": ModelOutput(sample_kind="atom")})
    st = out["mtt::aux::cutoff_stats"].block().values.detach().double().numpy()
    nl = sys_.get_neighbor_list(model.requested_neighbor_lists()[0])
    pairs = nl.samples.values[:, :2].numpy()
    d = torch.linalg.norm(nl.values.reshape(-1, 3).double(), dim=1).numpy()
    return float(out["energy"].block().values.sum()), st[:, 0], st[:, 1].astype(int), pairs, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--edges-dir", required=True)
    ap.add_argument("ids", nargs="+", type=int)
    a = ap.parse_args()
    from export_pytorch.export_pet_full import load_local_model

    model = load_local_model(a.ckpt)
    data = memmap(a.split)
    rng = np.random.default_rng(0)
    for s in a.ids:
        kept_c, r_c, full_c, E_c = mlipcpp_atoms(Path(a.edges_dir) / f"{s}.txt")
        E64, r64, k64, pairs, d = evaluate(model, data, s, torch.float64)
        E32, r32, k32, _, _ = evaluate(model, data, s, torch.float32)
        n = len(r64)
        print(f"\n=== structure {s}: {n} atoms, {len(d)} edges within the cutoff "
              f"(mlip.cpp full list {full_c.sum()}), pbc {bool((data['c'][s] != 0).any())}")
        print(f"E metatrain f64 {E64:.6f}  f32 {E32:.6f}  GraphModel CPU {E_c:.6f}  "
              f"(meV/atom vs f64: f32 {1e3 * (E32 - E64) / n:+.3f}, mlip.cpp {1e3 * (E_c - E64) / n:+.3f})")
        print(f"radius |diff| max: f32 {np.abs(r32 - r64).max():.3g} A, mlip.cpp {np.abs(r_c - r64).max():.3g} A; "
              f"kept-edge count differs on {(k32 != k64).sum()} (f32) / {(kept_c != k64).sum()} (mlip.cpp) atoms")
        m64 = d - 0.5 * (r64[pairs[:, 0]] + r64[pairs[:, 1]])
        for name, r in [("f32", r32), ("mlip.cpp", r_c)]:
            m = d - 0.5 * (r[pairs[:, 0]] + r[pairs[:, 1]])
            flip = np.flatnonzero((m <= 0) != (m64 <= 0))
            print(f"  {name}: {len(flip)} edges flip vs f64" + "".join(
                f"\n    edge {pairs[e, 0]}-{pairs[e, 1]} d {d[e]:.6f}: margin f64 {m64[e]:+.2e}, {name} {m[e]:+.2e}"
                for e in flip[:8]))
        near = np.argsort(np.abs(m64))[:3]
        print("  closest to the keep threshold (f64 margin):", ", ".join(f"{m64[e]:+.1e}" for e in near))
        for h in (1e-6, 1e-4):
            dx = torch.tensor(rng.normal(size=(n, 3)) * h)
            Eh = evaluate(model, data, s, torch.float64, dx)[0]
            print(f"  f64 energy moves {1e3 * (Eh - E64):+.4f} meV under a {h:g} A random displacement")


if __name__ == "__main__":
    main()
