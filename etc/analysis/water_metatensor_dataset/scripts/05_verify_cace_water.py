"""
Round-trip check for cace_water.zip: reload with metatrain's DiskDataset
reader and diff against the source xyz file read directly with ASE.
"""
from pathlib import Path

import ase.io
import numpy as np

from metatrain.utils.data import DiskDataset

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "cace_water.zip"
SRC = Path("/home/rumiants/pkgs/cace-lr-fit/fit-water/water.xyz")

dataset = DiskDataset(str(ZIP))
frames = ase.io.read(SRC, index=":")
print(f"DiskDataset: {len(dataset)} structures, source: {len(frames)} frames")
assert len(dataset) == len(frames)

max_pos_err = max_energy_err = max_force_err = max_cell_err = 0.0
for idx in range(len(dataset)):
    sample = dataset[idx]
    system = sample["system"]
    energy_tm = sample["energy"]
    atoms = frames[idx]

    pos = system.positions.numpy()
    cell = system.cell.numpy()
    e = energy_tm.block().values.item()
    grad_block = energy_tm.block().gradient("positions")
    forces = -grad_block.values.reshape(len(atoms), 3).numpy()

    max_pos_err = max(max_pos_err, np.abs(pos - atoms.get_positions()).max())
    max_cell_err = max(max_cell_err, np.abs(cell - atoms.cell[:]).max())
    max_energy_err = max(max_energy_err, abs(e - atoms.get_potential_energy()))
    max_force_err = max(max_force_err, np.abs(forces - atoms.arrays["force"]).max())

print(f"max |Δposition| = {max_pos_err:.3e} A")
print(f"max |Δcell|     = {max_cell_err:.3e} A")
print(f"max |Δenergy|   = {max_energy_err:.3e} eV")
print(f"max |Δforce|    = {max_force_err:.3e} eV/A")

assert max_pos_err < 1e-6
assert max_cell_err < 1e-6
assert max_energy_err < 1e-6
assert max_force_err < 1e-6
print("OK: cace_water.zip round-trips exactly against the source xyz.")
