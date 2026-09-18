"""Rattled bulk-silicon supercells for the heavy pipeline benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import ase.build
import numpy as np
from ase.io import write


def make_si_bulk(
    n_cells: int, n_structures: int, seed: int
) -> list:
    """Diamond Si, `n_cells`^3 conventional cells (3 -> 216 atoms)."""
    rng = np.random.default_rng(seed)
    primitive = ase.build.bulk("Si", "diamond", a=5.43, cubic=True)
    template = primitive.repeat((n_cells, n_cells, n_cells))
    frames = []
    for i in range(n_structures):
        atoms = template.copy()
        atoms.rattle(stdev=0.05, seed=int(rng.integers(0, 2**31)))
        atoms.info["energy"] = float(-5.43 * len(atoms) + rng.normal(0.0, 0.1))
        atoms.info["config_type"] = f"si_bulk_{len(atoms)}_{i}"
        frames.append(atoms)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-cells", type=int, default=3)
    parser.add_argument("--n-structures", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames = make_si_bulk(args.n_cells, args.n_structures, args.seed)
    write(args.out, frames, format="extxyz")
    print(f"wrote {len(frames)} x {len(frames[0])} atoms to {args.out}")


if __name__ == "__main__":
    main()
