"""
Convert a second, larger water dataset found on this shared filesystem into a
metatensor DiskDataset, alongside the DeePMD-kit-derived one built by
02_build_disk_dataset.py.

Source: /home/rumiants/pkgs/cace-lr-fit/fit-water/water.xyz
  - Same physical system as the DeePMD-kit dataset (64 H2O = 192 atoms, cubic
    periodic box, ~12.42 A edge), but 1593 frames instead of 320.
  - Used locally as CACE (Cheng group, https://github.com/BingqingCheng/cace)
    training data (see fit-cace-nnp.py in the same directory).
  - Read-only source: nothing under /home/rumiants is modified; only this
    converted copy is written, under this analysis directory.
  - License of the cace-lr-fit repo (per its README.md): CC BY-NC 4.0
    (non-commercial) -- carried over here, note it before reusing this
    converted copy commercially.
  - Per-atom forces are stored under the array key "force" (singular) in the
    source file, and energy is parsed by ASE into a SinglePointCalculator
    rather than atoms.info, hence get_potential_energy() below.
"""
from pathlib import Path

import ase.io
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import systems_to_torch

from metatrain.utils.data.writers import DiskDatasetWriter

SRC = Path("/home/rumiants/pkgs/cace-lr-fit/fit-water/water.xyz")
OUT = Path(__file__).resolve().parent.parent / "cace_water.zip"


def make_energy_tensormap(energy: float, forces) -> TensorMap:
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
        samples=Labels(["sample", "atom"], torch.tensor([[0, a] for a in range(n)])),
        components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    energy_block.add_gradient("positions", gradient_block)

    return TensorMap(keys=Labels.single(), blocks=[energy_block])


print(f"Reading {SRC} ...")
frames = ase.io.read(SRC, index=":")
print(f"Read {len(frames)} frames of {len(frames[0])} atoms each.")

if OUT.exists():
    OUT.unlink()

writer = DiskDatasetWriter(str(OUT))
for atoms in frames:
    energy = atoms.get_potential_energy()
    forces = atoms.arrays["force"]
    system = systems_to_torch(atoms, dtype=torch.float64)
    tensormap = make_energy_tensormap(float(energy), forces)
    writer.write([system], {"energy": tensormap})

writer.finish()
print(f"Wrote {len(frames)} structures to {OUT}")
