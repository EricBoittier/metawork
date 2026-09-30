"""Relax heme site SITEk (from the GROMACS groups, capped as metatomic-link-atoms does) with
several models and write the geometries: relax_sites.py GRO NDX SITE OUTDIR."""
import os, sys
import ase, ase.io
import numpy as np
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from metatomic.torch.ase_calculator import MetatomicCalculator

gro, ndx, site, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
os.makedirs(out, exist_ok=True)
lines = open(gro).read().splitlines(); n = int(lines[1]); at = lines[2:2 + n]
X = np.array([[float(l[20 + 8 * k:28 + 8 * k]) for k in range(3)] for l in at]) * 10
name = [l[10:15].strip() for l in at]
groups, cur = {}, None
for l in open(ndx):
    if l.startswith("["):
        cur = l.strip("[] \n"); groups[cur] = []
    else:
        groups[cur] += [int(t) - 1 for t in l.split()]
idx = groups[f"SITE{site}"]
cb = [k for k, i in enumerate(idx) if name[i] == "CB"][0]
ca = min((j for j in range(n) if name[j] == "CA" and j not in idx), key=lambda j: np.linalg.norm(X[j] - X[idx[cb]]))
d = X[ca] - X[idx[cb]]
pos = [X[i] for i in idx] + [X[idx[cb]] + d / np.linalg.norm(d) * 1.0]
sym = ["Fe" if name[i] == "FE" else name[i][0] for i in idx] + ["H"]
cluster = ase.Atoms(sym, positions=pos)
cluster.info["names"] = " ".join([name[i] for i in idx] + ["HL"])
ase.io.write(f"{out}/crystal.xyz", cluster)
MODELS = {"pet-mad-xs": ("pet-mad-xs-v1.6.0.pt", {}),
          "pet-omol-s-quintet": ("pet-omol-s-v1.0.0.pt", {"charge": -2, "spin_multiplicity": 5}),
          "pet-omol-s-singlet": ("pet-omol-s-v1.0.0.pt", {"charge": -2, "spin_multiplicity": 1})}
for tag, (path, info) in MODELS.items():
    a = cluster.copy(); a.info.update(info); a.calc = MetatomicCalculator(path, device="cuda")
    a.set_constraint(FixAtoms(indices=[cb, len(a) - 1]))
    BFGS(a, logfile=None).run(fmax=0.02, steps=600)
    a.calc = None; a.set_constraint()
    ase.io.write(f"{out}/{tag}.xyz", a)
    print(tag, "done")
