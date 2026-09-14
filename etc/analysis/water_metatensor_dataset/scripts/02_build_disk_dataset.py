"""
Convert the water dataset into metatrain's native metatensor `DiskDataset` format:
a zip archive with one folder per structure, each containing a serialized
`metatomic.torch.System` (system.mta) and a metatensor `TensorMap` (energy.mts)
holding the total energy plus a "positions" gradient block holding the forces
(stored as -forces, per the dE/dr = -F convention).

Reads the same raw DeePMD-kit .npy/.raw files as 01_build_xyz.py, so this script
can be run independently of the .xyz file.
"""
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import systems_to_torch

from metatrain.utils.data.writers import DiskDatasetWriter

RAW = Path(__file__).resolve().parent.parent / "raw_deepmd"
OUT = Path(__file__).resolve().parent.parent / "water_deepmd.zip"

type_map = RAW.joinpath("type_map.raw").read_text().split()
types = np.array([int(t) for t in RAW.joinpath("type.raw").read_text().split()])
symbols = [type_map[t] for t in types]
n_atoms = len(symbols)
assert n_atoms == 192, n_atoms


def make_energy_tensormap(energy: float, forces: np.ndarray) -> TensorMap:
    """One structure's energy target as a TensorMap, forces stored as a
    "positions" gradient block (values = -forces, since gradient = dE/dr = -F)."""
    forces_t = torch.as_tensor(forces, dtype=torch.float64)
    n = forces_t.shape[0]

    energy_block = TensorBlock(
        values=torch.tensor([[energy]], dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[0]])),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )

    gradient_block = TensorBlock(
        values=(-forces_t).reshape(n, 3, 1),
        samples=Labels(
            ["sample", "atom"],
            torch.tensor([[0, a] for a in range(n)]),
        ),
        components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    energy_block.add_gradient("positions", gradient_block)

    return TensorMap(keys=Labels.single(), blocks=[energy_block])


if OUT.exists():
    OUT.unlink()

writer = DiskDatasetWriter(str(OUT))

n_written = 0
for set_idx in range(4):
    set_dir = RAW / f"set_{set_idx}"
    box = np.load(set_dir / "box.npy")
    coord = np.load(set_dir / "coord.npy")
    energy = np.load(set_dir / "energy.npy")
    force = np.load(set_dir / "force.npy")

    n_frames = box.shape[0]
    for i in range(n_frames):
        cell = box[i].reshape(3, 3)
        pos = coord[i].reshape(n_atoms, 3)
        frc = force[i].reshape(n_atoms, 3)

        atoms = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)
        system = systems_to_torch(atoms, dtype=torch.float64)
        tensormap = make_energy_tensormap(float(energy[i]), frc)

        writer.write([system], {"energy": tensormap})
        n_written += 1

writer.finish()
print(f"Wrote {n_written} structures to {OUT}")
