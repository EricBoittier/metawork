"""
Build an ASE-readable extxyz file from the raw DeePMD-kit water dataset.

Source: deepmodeling/deepmd-kit, examples/water/data/{data_0..data_3}
        (bulk liquid water, 64 H2O molecules = 192 atoms/frame, cubic PBC box,
        PBE-DFT energies [eV] and forces [eV/Angstrom], 80 frames per shard).
"""
from pathlib import Path

import ase.io
import numpy as np
from ase import Atoms

RAW = Path(__file__).resolve().parent.parent / "raw_deepmd"
OUT = Path(__file__).resolve().parent.parent / "water_deepmd.xyz"

type_map = RAW.joinpath("type_map.raw").read_text().split()
types = np.array([int(t) for t in RAW.joinpath("type.raw").read_text().split()])
symbols = [type_map[t] for t in types]
n_atoms = len(symbols)
assert n_atoms == 192, n_atoms

frames = []
for set_idx in range(4):
    set_dir = RAW / f"set_{set_idx}"
    box = np.load(set_dir / "box.npy")        # (n_frames, 9)
    coord = np.load(set_dir / "coord.npy")    # (n_frames, n_atoms*3)
    energy = np.load(set_dir / "energy.npy")  # (n_frames,)
    force = np.load(set_dir / "force.npy")    # (n_frames, n_atoms*3)

    n_frames = box.shape[0]
    for i in range(n_frames):
        cell = box[i].reshape(3, 3)
        pos = coord[i].reshape(n_atoms, 3)
        frc = force[i].reshape(n_atoms, 3)
        atoms = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)
        atoms.info["energy"] = float(energy[i])
        atoms.info["config_type"] = f"deepmd_water_set_{set_idx}"
        atoms.arrays["forces"] = frc
        frames.append(atoms)

print(f"Built {len(frames)} frames of {n_atoms} atoms each "
      f"({n_atoms // 3} H2O molecules).")
ase.io.write(OUT, frames)
print(f"Wrote {OUT}")
