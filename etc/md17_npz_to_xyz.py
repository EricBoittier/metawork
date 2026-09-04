#!/usr/bin/env python3
"""Convert a raw MD17/MD22-style .npz trajectory (as downloaded from sGDML) into
an extended-XYZ file that ASE / metatensor / metatomic / metatrain can read
directly, with energies and forces converted from kcal/mol (+ kcal/mol/A) to
eV (+ eV/A) -- the convention used throughout the metatensor ecosystem
examples (see e.g. metatrain's 0-beginner/01-data_preparation.py).

Usage:
    md17_npz_to_xyz.py INPUT.npz OUTPUT.xyz [--n-samples N] [--seed SEED]

By default this writes a random subsample of 1000 frames (these
trajectories have 100k-1M near-duplicate frames; a subsample is plenty for
training/testing). Pass --n-samples all to write every frame.
"""

import argparse
import sys

import numpy as np
from ase import Atoms
from ase.io import write
from ase.units import kcal, mol

KCAL_MOL_TO_EV = kcal / mol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="input .npz file (MD17/MD22 format: R, E, F, z)")
    parser.add_argument("output", help="output .xyz file")
    parser.add_argument(
        "--n-samples",
        default="1000",
        help="number of frames to write, or 'all' (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="subsampling RNG seed")
    args = parser.parse_args()

    data = np.load(args.input)
    positions = data["R"]  # (n_frames, n_atoms, 3), Angstrom
    energies = data["E"].reshape(-1)  # (n_frames,), kcal/mol
    forces = data["F"]  # (n_frames, n_atoms, 3), kcal/mol/Angstrom
    numbers = data["z"]  # (n_atoms,)

    n_frames = positions.shape[0]
    if args.n_samples == "all":
        indices = np.arange(n_frames)
    else:
        n_samples = min(int(args.n_samples), n_frames)
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(n_frames, size=n_samples, replace=False))

    frames = []
    for i in indices:
        atoms = Atoms(numbers=numbers, positions=positions[i])
        atoms.info["energy"] = float(energies[i]) * KCAL_MOL_TO_EV
        atoms.arrays["forces"] = forces[i] * KCAL_MOL_TO_EV
        frames.append(atoms)

    write(args.output, frames)
    print(
        f"wrote {len(frames)}/{n_frames} frames from {args.input} to {args.output} "
        f"(energy in eV, forces in eV/A)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
