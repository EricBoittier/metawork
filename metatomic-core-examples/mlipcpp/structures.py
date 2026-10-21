"""Write the test structures used by compare-torch.sh to <dir>/*.xyz."""
import sys

from ase.build import bulk, molecule
from ase.io import write

out = sys.argv[1]
si = bulk("Si", "diamond", a=5.43, cubic=True)
si.rattle(0.05, seed=1)
slab = si.copy()
slab.pbc = [True, True, False]
slab.cell[2] = 0
nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(2)
nacl.rattle(0.05, seed=2)
fe = bulk("Fe", "bcc", a=2.87, cubic=True).repeat(3)
fe.rattle(0.05, seed=3)
cluster = si.copy()
cluster.pbc = False
cluster.cell[:] = 0
for name, atoms in dict(si=si, slab=slab, nacl=nacl, fe=fe, cluster=cluster, water=molecule("H2O")).items():
    write(f"{out}/{name}.xyz", atoms, format="extxyz")
