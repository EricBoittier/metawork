#!/usr/bin/env python3
"""Download the PhysNet SN2 reactions set and write an extended XYZ.

Source: https://zenodo.org/records/2605341
Unke & Meuwly, *PhysNet* (arXiv:1902.08408); DOI 10.5281/zenodo.2605341.

The npz is 119 MB and ~452k structures. Default is a 1000-structure
random subsample. Energies are atomization energies (eV), forces eV/Å,
dipoles e·Å wrt the origin.

The XYZ key is ``dipole_moment`` (not ASE-reserved ``dipole``).

Examples:

    python etc/sn2_zenodo/convert.py --n-samples 1000
    python etc/sn2_zenodo/convert.py --n-samples all --data-dir ~/data/sn2
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols

ZENODO = "https://zenodo.org/records/2605341/files"
NPZ_NAME = "sn2_reactions.npz"


def _download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    url = f"{ZENODO}/{NPZ_NAME}?download=1"
    print(f"downloading {url} -> {dest}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "metawork-sn2"})
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(req) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out)
    tmp.rename(dest)
    return dest


def npz_to_atoms(data: np.lib.npyio.NpzFile, index: int) -> Atoms:
    n_atoms = int(data["N"][index])
    atoms = Atoms(
        numbers=np.asarray(data["Z"][index, :n_atoms], dtype=int),
        positions=np.asarray(data["R"][index, :n_atoms], dtype=float),
    )
    atoms.info["energy"] = float(data["E"][index])
    atoms.info["charge"] = float(data["Q"][index])
    atoms.info["dipole_moment"] = np.asarray(data["D"][index], dtype=float).reshape(3)
    atoms.arrays["forces"] = np.asarray(data["F"][index, :n_atoms], dtype=float)
    return atoms


def write_extxyz(data, indices, path: Path) -> None:
    """Write an ASE-compatible extxyz without building all frames in memory."""
    with path.open("w") as handle:
        for index in indices:
            index = int(index)
            n_atoms = int(data["N"][index])
            numbers = np.asarray(data["Z"][index, :n_atoms], dtype=int)
            positions = np.asarray(data["R"][index, :n_atoms], dtype=float)
            forces = np.asarray(data["F"][index, :n_atoms], dtype=float)
            energy = float(data["E"][index])
            charge = float(data["Q"][index])
            dipole = np.asarray(data["D"][index], dtype=float).reshape(3)
            handle.write(f"{n_atoms}\n")
            handle.write(
                "Properties=species:S:1:pos:R:3:forces:R:3 "
                f"energy={energy} charge={charge} "
                f'dipole_moment="{dipole[0]} {dipole[1]} {dipole[2]}" '
                'pbc="F F F"\n'
            )
            for number, xyz, force in zip(numbers, positions, forces, strict=True):
                symbol = chemical_symbols[int(number)]
                handle.write(
                    f"{symbol:2s} "
                    f"{xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f} "
                    f"{force[0]:16.8f} {force[1]:16.8f} {force[2]:16.8f}\n"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "data" / "sn2",
        help="download and write here (default: ~/data/sn2)",
    )
    parser.add_argument(
        "--n-samples",
        default="1000",
        help="random subsample size, or 'all' (default: 1000)",
    )
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=5,
        help="drop fragments smaller than this (default: 5, keeps CH3X / SN2)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = _download(args.data_dir / NPZ_NAME)
    loaded = np.load(npz_path)
    data = {key: np.asarray(loaded[key]) for key in ("N", "Z", "R", "F", "E", "Q", "D")}
    loaded.close()
    n_atoms = np.asarray(data["N"], dtype=int)
    eligible = np.flatnonzero(n_atoms >= int(args.min_atoms))
    n_total = int(n_atoms.shape[0])
    if str(args.n_samples) == "all":
        indices = eligible
    else:
        n_keep = min(int(args.n_samples), int(eligible.shape[0]))
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(eligible, size=n_keep, replace=False))

    xyz_path = args.data_dir / "sn2.xyz"
    write_extxyz(data, indices, xyz_path)
    print(
        f"wrote {len(indices)} structures "
        f"(from {len(eligible)} with N>= {args.min_atoms}, "
        f"{n_total} total) -> {xyz_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
