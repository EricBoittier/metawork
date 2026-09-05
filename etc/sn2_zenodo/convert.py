#!/usr/bin/env python3
"""Download the PhysNet SN2 reactions set and write an extended XYZ.

Source: https://zenodo.org/records/2605341
Unke & Meuwly, *PhysNet* (arXiv:1902.08408); DOI 10.5281/zenodo.2605341.

The npz is 119 MB and ~452k structures. Default is a random subsample
so a first ``mtt train`` stays small. Energies are atomization energies
(eV), forces eV/Å, dipoles e·Å wrt the origin.

The XYZ key is ``dipole_moment`` (not ASE-reserved ``dipole``).

Examples:

    python etc/sn2_zenodo/convert.py --n-samples 200
    python etc/sn2_zenodo/convert.py --n-samples all --data-dir ~/data/sn2
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

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
        default="200",
        help="random subsample size, or 'all' (default: 200)",
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
    data = np.load(npz_path)
    n_atoms = np.asarray(data["N"], dtype=int)
    eligible = np.flatnonzero(n_atoms >= int(args.min_atoms))
    n_total = int(n_atoms.shape[0])
    if str(args.n_samples) == "all":
        indices = eligible
    else:
        n_keep = min(int(args.n_samples), int(eligible.shape[0]))
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(eligible, size=n_keep, replace=False))

    frames = [npz_to_atoms(data, int(i)) for i in indices]
    xyz_path = args.data_dir / "sn2.xyz"
    write(xyz_path, frames, format="extxyz")
    print(
        f"wrote {len(frames)} structures "
        f"(from {len(eligible)} with N>= {args.min_atoms}, "
        f"{n_total} total) -> {xyz_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
