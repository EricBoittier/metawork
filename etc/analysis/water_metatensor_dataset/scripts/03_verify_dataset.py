"""
Round-trip check: reload water_deepmd.zip with metatrain's own DiskDataset
reader and confirm energies/forces/positions match the source .npy arrays.
"""
from pathlib import Path

import numpy as np

from metatrain.utils.data import read_systems, read_targets, DiskDataset

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "water_deepmd.zip"
RAW = ROOT / "raw_deepmd"

dataset = DiskDataset(str(ZIP))
print(f"DiskDataset loaded: {len(dataset)} structures")

# Reference arrays, in the same frame order as 02_build_disk_dataset.py
type_map = RAW.joinpath("type_map.raw").read_text().split()
types = np.array([int(t) for t in RAW.joinpath("type.raw").read_text().split()])
n_atoms = len(types)

ref_energy, ref_force, ref_pos, ref_cell = [], [], [], []
for set_idx in range(4):
    set_dir = RAW / f"set_{set_idx}"
    box = np.load(set_dir / "box.npy")
    coord = np.load(set_dir / "coord.npy")
    energy = np.load(set_dir / "energy.npy")
    force = np.load(set_dir / "force.npy")
    for i in range(box.shape[0]):
        ref_energy.append(energy[i])
        ref_force.append(force[i].reshape(n_atoms, 3))
        ref_pos.append(coord[i].reshape(n_atoms, 3))
        ref_cell.append(box[i].reshape(3, 3))

assert len(dataset) == len(ref_energy)

max_pos_err = max_energy_err = max_force_err = max_cell_err = 0.0
for idx in range(len(dataset)):
    sample = dataset[idx]
    system = sample["system"]
    energy_tm = sample["energy"]

    pos = system.positions.numpy()
    cell = system.cell.numpy()
    e = energy_tm.block().values.item()
    # forces = -gradient
    grad_block = energy_tm.block().gradient("positions")
    forces = -grad_block.values.reshape(n_atoms, 3).numpy()

    max_pos_err = max(max_pos_err, np.abs(pos - ref_pos[idx]).max())
    max_cell_err = max(max_cell_err, np.abs(cell - ref_cell[idx]).max())
    max_energy_err = max(max_energy_err, abs(e - ref_energy[idx]))
    max_force_err = max(max_force_err, np.abs(forces - ref_force[idx]).max())

print(f"max |Δposition| = {max_pos_err:.3e} A")
print(f"max |Δcell|     = {max_cell_err:.3e} A")
print(f"max |Δenergy|   = {max_energy_err:.3e} eV")
print(f"max |Δforce|    = {max_force_err:.3e} eV/A")

assert max_pos_err < 1e-6
assert max_cell_err < 1e-6
assert max_energy_err < 1e-6
assert max_force_err < 1e-6
print("OK: DiskDataset round-trips exactly against the source data.")
