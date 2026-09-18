"""Synthetic dataset with OMol-like structure-size diversity.

OMol25 (Meta/FAIR's Open Molecules dataset) is not available here -- it is a
large, gated Hugging Face dataset. This generates a size-diverse proxy
instead: small G2-database molecules, mid-size rattled clusters cut from
bulk, and larger rattled bulk supercells, spanning roughly 5-300 atoms per
structure. The point is the *spread*, to exercise atoms-per-batch batching
the way a real OMol-scale dataset would -- not chemical realism.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ase.build
import numpy as np
from ase.io import write

# A range of G2-database molecules already bundled with ASE, small end of
# the size spread (2-24 atoms).
SMALL_MOLECULES = [
    "H2", "H2O", "CH4", "NH3", "C2H2", "C2H4", "C2H6", "CH3OH", "C6H6",
    "C3H8", "CH3CHO", "CH3COOH", "C5H5N", "cyclobutane", "bicyclobutane",
]

# Bulk lattices (element, structure, lattice constant) to build mid/large
# rattled supercells from, repeated 1x1x1 through 4x4x4.
BULK_TEMPLATES = [
    ("Si", "diamond", 5.43),
    ("Cu", "fcc", 3.615),
    ("Fe", "bcc", 2.87),
    ("NaCl", "rocksalt", 5.64),
]


def _small_molecule(rng: np.random.Generator):
    name = SMALL_MOLECULES[rng.integers(0, len(SMALL_MOLECULES))]
    atoms = ase.build.molecule(name)
    atoms.rattle(stdev=0.03, seed=int(rng.integers(0, 2**31)))
    return atoms


def _bulk_cluster(rng: np.random.Generator, max_cells: int):
    element, structure, a = BULK_TEMPLATES[rng.integers(0, len(BULK_TEMPLATES))]
    n_cells = int(rng.integers(1, max_cells + 1))
    if structure == "rocksalt":
        primitive = ase.build.bulk(element, structure, a=a, cubic=True)
    else:
        primitive = ase.build.bulk(element, structure, a=a, cubic=True)
    atoms = primitive.repeat((n_cells, n_cells, n_cells))
    atoms.rattle(stdev=0.05, seed=int(rng.integers(0, 2**31)))
    return atoms


def make_omol_proxy(n_structures: int, max_cells: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_structures):
        # roughly a third small molecules, two-thirds bulk-derived clusters
        # of varying repeat count -- the actual size spread that matters.
        if rng.random() < 0.3:
            atoms = _small_molecule(rng)
            tier = "small"
        else:
            atoms = _bulk_cluster(rng, max_cells)
            tier = "bulk"
        atoms.info["energy"] = float(-4.0 * len(atoms) + rng.normal(0.0, 0.2))
        atoms.info["config_type"] = f"omol_proxy_{tier}_{len(atoms)}_{i}"
        frames.append(atoms)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-structures", type=int, default=200)
    parser.add_argument(
        "--max-cells",
        type=int,
        default=4,
        help="max bulk repeat count per axis (1 -> primitive cell, "
        "4 -> up to 4^3 cells)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames = make_omol_proxy(args.n_structures, args.max_cells, args.seed)
    write(args.out, frames, format="extxyz")
    sizes = sorted(len(f) for f in frames)
    print(
        f"wrote {len(frames)} structures to {args.out}, "
        f"sizes {sizes[0]}-{sizes[-1]} atoms "
        f"(median {sizes[len(sizes) // 2]})"
    )


if __name__ == "__main__":
    main()
