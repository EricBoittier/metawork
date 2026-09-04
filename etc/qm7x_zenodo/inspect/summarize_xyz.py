#!/usr/bin/env python3
"""Summarize a converted QM7-X extended XYZ (and optional sidecar .mts)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz", type=Path)
    parser.add_argument(
        "--mts",
        type=Path,
        default=None,
        help="polarizability sidecar (default: sibling polarizability_spherical.mts)",
    )
    args = parser.parse_args()
    frames = read(args.xyz, index=":")
    if not isinstance(frames, list):
        frames = [frames]

    elements: Counter[str] = Counter()
    n_atoms = []
    energies = []
    info_keys: Counter[str] = Counter()
    array_keys: Counter[str] = Counter()
    for atoms in frames:
        elements.update(atoms.get_chemical_symbols())
        n_atoms.append(len(atoms))
        info_keys.update(atoms.info.keys())
        array_keys.update(atoms.arrays.keys())
        if "energy" in atoms.info:
            energies.append(float(atoms.info["energy"]))

    print(f"{args.xyz}: {len(frames)} structures")
    print(f"  atoms/structure  min={min(n_atoms)} max={max(n_atoms)} mean={np.mean(n_atoms):.1f}")
    print(f"  elements         {dict(sorted(elements.items()))}")
    print(f"  info keys        {dict(info_keys)}")
    print(f"  arrays keys      {dict(array_keys)}")
    if energies:
        arr = np.asarray(energies)
        print(
            f"  energy / eV      min={arr.min():.4f} max={arr.max():.4f} "
            f"mean={arr.mean():.4f}"
        )

    mts = args.mts or args.xyz.with_name("polarizability_spherical.mts")
    if mts.exists():
        from metatensor.torch import TensorMap

        tmap = TensorMap.load(str(mts))
        print(f"{mts}: keys={tmap.keys}")
        for key, block in tmap.items():
            print(f"  {key}: values={tuple(block.values.shape)}")
    else:
        print(f"no sidecar {mts}")


if __name__ == "__main__":
    main()
