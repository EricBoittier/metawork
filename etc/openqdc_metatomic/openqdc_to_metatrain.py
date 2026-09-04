#!/usr/bin/env python3
"""Convert an OpenQDC dataset into a format metatrain can train on.

OpenQDC already memory-maps geometries, energies and forces. Iterating
``dataset.as_iter(atoms=True)`` builds an ASE ``Atoms`` object per structure
and is the slow path — and the resulting objects store labels under OpenQDC
keys (``info["energies"]``, ``info["forces"]``) that metatrain will not
pick up. This script indexes the dataset as arrays, remaps labels to the
metatensor/metatrain conventions, and writes one of:

* ``xyz``     — extended XYZ, fine for a few thousand structures
* ``zip``     — ``DiskDataset`` (``*.mta`` systems + ``*.mts`` targets)
* ``memmap``  — directory of memory-mapped arrays (recommended for QM7-X)

Examples:

    openqdc_to_metatrain.py --dataset QM7X --n-samples 1000 --format xyz
    openqdc_to_metatrain.py --dataset QM7X --n-samples 10000 --format zip
    openqdc_to_metatrain.py --dataset QM7X --n-samples all --format memmap
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import systems_to_torch

from metatrain.utils.data.writers import DiskDatasetWriter, MemmapWriter


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value)
    return np.asarray(value)


def _load_dataset(name: str, energy_unit: str, distance_unit: str, energy_type: str):
    module = importlib.import_module("openqdc.datasets")
    cls = getattr(module, name)
    return cls(
        energy_unit=energy_unit,
        distance_unit=distance_unit,
        array_format="numpy",
        energy_type=energy_type,
    )


def _indices(dataset, n_samples: str, seed: int) -> np.ndarray:
    n_frames = len(dataset)
    if n_samples == "all":
        return np.arange(n_frames)
    n = min(int(n_samples), n_frames)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_frames, size=n, replace=False))


def _has_forces(dataset, energy_method: int) -> bool:
    mask = getattr(dataset, "force_mask", None)
    if mask is None:
        return False
    if energy_method >= len(mask):
        return False
    return bool(mask[energy_method])


def _force_method_index(dataset, energy_method: int) -> int:
    """Map an energy-method index onto the (usually shorter) force axis."""
    mask = list(getattr(dataset, "force_mask", []))
    return int(sum(mask[:energy_method]))


def _entry_arrays(entry: dict, energy_method: int, force_method: int | None):
    positions = _as_numpy(entry["positions"])
    numbers = _as_numpy(entry["atomic_numbers"]).astype(np.int32)
    energy = float(_as_numpy(entry["energies"]).reshape(-1)[energy_method])
    charges = _as_numpy(entry["charges"]).astype(np.int32)
    forces = None
    if force_method is not None and "forces" in entry:
        forces = _as_numpy(entry["forces"])[:, :, force_method]
    return numbers, positions, charges, energy, forces


def _energy_tensormap(
    energy: float,
    forces: np.ndarray | None,
    system_index: int,
) -> TensorMap:
    block = TensorBlock(
        values=torch.tensor([[energy]], dtype=torch.float64),
        samples=Labels("system", torch.tensor([[system_index]])),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    if forces is not None:
        n_atoms = forces.shape[0]
        block.add_gradient(
            "positions",
            TensorBlock(
                values=-torch.tensor(forces, dtype=torch.float64).unsqueeze(-1),
                samples=Labels(
                    names=["sample", "atom"],
                    values=torch.tensor([[0, i] for i in range(n_atoms)]),
                ),
                components=[Labels("xyz", torch.tensor([[0], [1], [2]]))],
                properties=Labels("energy", torch.tensor([[0]])),
            ),
        )
    return TensorMap(Labels.single(), [block])


def write_xyz(dataset, indices, energy_method, out_path: Path) -> None:
    force_method = (
        _force_method_index(dataset, energy_method)
        if _has_forces(dataset, energy_method)
        else None
    )
    frames = []
    for idx in indices:
        numbers, positions, charges, energy, forces = _entry_arrays(
            dataset[int(idx)], energy_method, force_method
        )
        atoms = Atoms(numbers=numbers, positions=positions, charges=charges)
        atoms.info["energy"] = energy
        if forces is not None:
            atoms.arrays["forces"] = forces
        frames.append(atoms)
    write(out_path, frames)


def write_metatrain(
    dataset,
    indices,
    energy_method,
    writer,
) -> None:
    force_method = (
        _force_method_index(dataset, energy_method)
        if _has_forces(dataset, energy_method)
        else None
    )
    for idx in indices:
        numbers, positions, charges, energy, forces = _entry_arrays(
            dataset[int(idx)], energy_method, force_method
        )
        atoms = Atoms(numbers=numbers, positions=positions, charges=charges)
        system = systems_to_torch(atoms, dtype=torch.float64)
        writer.write(
            [system],
            {"energy": _energy_tensormap(energy, forces, system_index=0)},
        )
    writer.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="QM7X", help="OpenQDC class name")
    parser.add_argument(
        "--format",
        choices=("xyz", "zip", "memmap"),
        default="xyz",
        help="metatrain dataset format to write",
    )
    parser.add_argument(
        "--n-samples",
        default="1000",
        help="number of structures, or 'all' (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--energy-method", type=int, default=0)
    parser.add_argument("--energy-unit", default="ev")
    parser.add_argument("--distance-unit", default="ang")
    parser.add_argument(
        "--energy-type",
        default="formation",
        choices=("formation", "regression", "null"),
        help="OpenQDC isolated-atom subtraction (default: formation)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output path (default: <dataset>.xyz / .zip / _memmap/)",
    )
    args = parser.parse_args()

    dataset = _load_dataset(
        args.dataset, args.energy_unit, args.distance_unit, args.energy_type
    )
    print(
        f"loaded {args.dataset}: {len(dataset)} structures, "
        f"methods={dataset.energy_methods}, "
        f"energy_unit={args.energy_unit}, energy_type={args.energy_type}",
        flush=True,
    )
    indices = _indices(dataset, args.n_samples, args.seed)
    stem = args.dataset.lower()
    if args.format == "xyz":
        out = Path(args.output or f"{stem}.xyz")
        write_xyz(dataset, indices, args.energy_method, out)
    elif args.format == "zip":
        out = Path(args.output or f"{stem}.zip")
        write_metatrain(
            dataset, indices, args.energy_method, DiskDatasetWriter(out)
        )
    else:
        out = Path(args.output or f"{stem}_memmap")
        write_metatrain(
            dataset, indices, args.energy_method, MemmapWriter(out)
        )
    print(f"wrote {len(indices)}/{len(dataset)} structures to {out}")


if __name__ == "__main__":
    main()
