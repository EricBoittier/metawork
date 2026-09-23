"""Check the `.mta` files written by extxyz-gz-to-mta against ASE's reading of the source file.

usage: python check_mta.py <input.extxyz[.gz]> <directory with <frame index>.mta files>
"""

import itertools
import sys
from pathlib import Path

import ase.io
import metatomic.torch as mta
import numpy as np

source, directory = sys.argv[1], Path(sys.argv[2])
saved = sorted(directory.glob("*.mta"), key=lambda p: int(p.stem))
# iread stops early, while ase.io.read(index=":n") indexes the whole file first
frames = list(itertools.islice(ase.io.iread(source, format="extxyz"), int(saved[-1].stem) + 1))
for path in saved:
    system, atoms = mta.load_system(str(path)), frames[int(path.stem)]
    assert system.types.tolist() == atoms.numbers.tolist(), path
    assert np.allclose(system.positions.numpy(), atoms.positions), path
    assert system.pbc.tolist() == atoms.pbc.tolist(), path
    # metatomic stores a zero cell vector along non-periodic directions
    assert np.allclose(system.cell.numpy(), atoms.cell.array * atoms.pbc[:, None]), path
n_periodic = sum(frames[int(p.stem)].pbc.all() for p in saved)
print(f"{len(saved)} systems match ASE ({n_periodic} periodic)")
