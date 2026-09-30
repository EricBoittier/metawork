"""PET-OMol energy of each ML site from the GROMACS site groups, capped as metatomic-link-atoms does.

ref_sites.py GRO NDX MODEL [CHARGE SPIN]: every SITEk group of NDX as one ASE system, with a
hydrogen 0.1 nm from each boundary CB toward its CA, charge and spin multiplicity per site.
"""
import sys
import ase
import numpy as np
from metatomic.torch.ase_calculator import MetatomicCalculator

gro, ndx, model = sys.argv[1:4]
charge = float(sys.argv[4]) if len(sys.argv) > 4 else -2
spin = int(sys.argv[5]) if len(sys.argv) > 5 else 5
lines = open(gro).read().splitlines(); n = int(lines[1]); at = lines[2:2 + n]
X = np.array([[float(l[20 + 8 * k:28 + 8 * k]) for k in range(3)] for l in at]) * 10
name = [l[10:15].strip() for l in at]
groups, cur = {}, None
for l in open(ndx):
    if l.startswith("["):
        cur = l.strip("[] \n"); groups[cur] = []
    else:
        groups[cur] += [int(t) - 1 for t in l.split()]
calc = MetatomicCalculator(model, device="cuda")
total, sites = 0.0, sorted(k for k in groups if k.startswith("SITE"))
for s in sites:
    idx = groups[s]
    pos, sym = [X[i] for i in idx], ["Fe" if name[i] == "FE" else name[i][0] for i in idx]
    for cb in (i for i in idx if name[i] == "CB"):
        ca = min((j for j in range(n) if name[j] == "CA" and j not in idx), key=lambda j: np.linalg.norm(X[j] - X[cb]))
        d = X[ca] - X[cb]
        pos.append(X[cb] + d / np.linalg.norm(d) * 1.0); sym.append("H")
    a = ase.Atoms(sym, positions=pos)
    a.info.update(charge=charge, spin_multiplicity=spin); a.calc = calc
    e = a.get_potential_energy() * 96.4853; total += e
    print(f"{s}: {len(a)} atoms, E = {e:.3f} kJ/mol")
print(f"total {total:.3f} kJ/mol")
