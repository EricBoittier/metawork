"""Compare the PyTorch export of a checkpoint with plugin results.

usage: torch_eval.py <model.pt> <plugin.npz> <xyz>...
"""
import sys

import ase.io
import numpy as np
from metatomic.torch.ase_calculator import MetatomicCalculator

pt, npz, *files = sys.argv[1:]
plugin = np.load(npz)
conservative = MetatomicCalculator(pt, device="cpu")
direct = MetatomicCalculator(pt, device="cpu", non_conservative="forces")

# the ASE calculator removes the net non-conservative force, the plugin returns
# the raw model output: compare with the plugin's mean removed (dFnc column)
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
print(f"{'system':8} {'n':>3} {'E torch':>12} {'E plugin':>12} {'dE/atom meV':>11} "
      f"{'F rms':>7} {'dF rms':>7} {'dFnc rms':>8} {'dvirial':>8}")
for name in files:
    key = name.split("/")[-1].removesuffix(".xyz")
    atoms = ase.io.read(name)
    atoms.calc = conservative
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    virial = None
    if atoms.pbc.all():
        virial = atoms.get_stress(voigt=False) * atoms.get_volume()  # dE/de = V sigma
    atoms.calc = direct
    fnc = atoms.get_forces()

    n = len(atoms)
    dv = rms(virial - plugin[f"{key}/virial"]) if virial is not None else float("nan")
    print(f"{key:8} {n:3d} {e:12.5f} {plugin[f'{key}/energy']:12.5f} "
          f"{1000 * (e - plugin[f'{key}/energy']) / n:11.3f} {rms(f):7.3f} "
          f"{rms(f - plugin[f'{key}/forces']):7.4f} {rms(fnc - plugin[f'{key}/nc_forces'] + plugin[f'{key}/nc_forces'].mean(0)):8.4f} {dv:8.4f}")
