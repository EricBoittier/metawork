"""Compare PET-MAD sizes on one heme site of deoxy hemoglobin.

site_test.py [GRO] [SITE]: the heme of the SITE-th iron (0-3) plus the side chain of its
proximal histidine from CB, capped with a hydrogen 0.1 nm from CB toward CA (as
metatomic-link-atoms does). For each model: forces at the input geometry, then a vacuum
relaxation with CB and the cap fixed, and the iron geometry against the crystal (1A3N).
"""

import sys

import ase
import numpy as np
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from metatomic.torch.ase_calculator import MetatomicCalculator

gro = sys.argv[1] if len(sys.argv) > 1 else "../hb.gro"
site = int(sys.argv[2]) if len(sys.argv) > 2 else 0
# name -> (model file, atoms.info); PET-OMol takes the total charge and spin multiplicity,
# the heme dianion (two propionates) with the neutral His side chain is -2
MODELS = {s: (f"pet-mad-{s}-v1.6.0.pt", {}) for s in ("xs", "s", "m")}
OMOL = {f"omol-{s}-2S+1={m}": (f"pet-omol-{s}-v1.0.0.pt", {"charge": -2, "spin_multiplicity": m})
        for s in ("s", "m", "l") for m in (5, 1)}
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None

lines = open(gro).read().splitlines()
n = int(lines[1])
atoms = lines[2 : 2 + n]
name = lambda i: atoms[i][10:15].strip()
resn = lambda i: atoms[i][5:10].strip()
resid = lambda i: int(atoms[i][:5])
X = lambda i: np.array([float(atoms[i][20 + 8 * k : 28 + 8 * k]) for k in range(3)]) * 10  # Angstrom

fe = [i for i in range(n) if name(i) == "FE"][site]
heme = [i for i in range(n) if resn(i) == "HEM" and resid(i) == resid(fe) and abs(i - fe) < 80]
# proximal His: the NE2 closest to this iron
ne2 = min((np.linalg.norm(X(j) - X(fe)), j) for j in range(n) if name(j) == "NE2" and resn(j) == "HIS")[1]
his = [j for j in range(ne2 - 15, ne2 + 10) if resid(j) == resid(ne2) and resn(j) == "HIS"]
side = [j for j in his if name(j) not in ("N", "HN", "CA", "HA", "C", "O")]
ca, cb = [j for j in his if name(j) == "CA"][0], [j for j in his if name(j) == "CB"][0]

idx = heme + side
el = lambda nm: "Fe" if nm == "FE" else nm[0]
pos = [X(i) for i in idx] + [X(cb) + (X(ca) - X(cb)) / np.linalg.norm(X(ca) - X(cb)) * 1.0]
cluster = ase.Atoms([el(name(i)) for i in idx] + ["H"], positions=pos)
at = {name(i): k for k, i in enumerate(idx)}
cap = len(idx)


def geometry(a):
    p = a.get_positions()
    fe_, np4 = p[at["FE"]], np.array([p[at[k]] for k in ("NA", "NB", "NC", "ND")])
    c = np4.mean(0)
    normal = np.linalg.svd(np4 - c)[2][-1]
    if normal @ (p[at["NE2"]] - c) < 0:
        normal = -normal
    return {
        "Fe-NE2": np.linalg.norm(fe_ - p[at["NE2"]]) / 10,
        "Fe-Np": np.linalg.norm(np4 - fe_, axis=1).mean() / 10,
        "Fe-plane": (fe_ - c) @ normal / 10,  # toward the histidine is positive
    }


print(f"site {site}: heme {resid(fe)} + His{resid(ne2)} side chain, {len(cluster)} atoms")
ref = geometry(cluster)
print("crystal  " + "  ".join(f"{k} {v:.3f} nm" for k, v in ref.items()))

forces = {}
for size, (path, info) in {**MODELS, **OMOL}.items():
    if only and size not in only:
        continue
    calc = MetatomicCalculator(path, device="cuda")
    a = cluster.copy()
    a.info.update(info)
    a.calc = calc
    forces[size] = a.get_forces()
    a.set_constraint(FixAtoms(indices=[at["CB"], cap]))
    BFGS(a, logfile=None).run(fmax=0.02, steps=600)
    g = geometry(a)
    print(f"{size:14s} relaxed  " + "  ".join(f"{k} {v:.3f} nm" for k, v in g.items()))

EV_A = 964.853  # eV/Angstrom -> kJ/mol/nm


def compare(ref, keys):
    """Force differences against the largest model of a family, at the input geometry."""
    if ref not in forces:
        return
    f0 = forces[ref]
    print(f"force scale ({ref}): RMS |F| {np.sqrt((f0**2).sum(1).mean()) * EV_A:.0f} kJ/mol/nm")
    for k in [k for k in keys if k in forces and k != ref]:
        d = np.linalg.norm(forces[k] - f0, axis=1) * EV_A
        print(f"forces {k} vs {ref}: RMS {np.sqrt((d**2).mean()):.0f}, Fe {d[at['FE']]:.0f}, "
              f"NE2 {d[at['NE2']]:.0f}, max {d.max():.0f} kJ/mol/nm")


compare("m", ["xs", "s"])
for m in (5, 1):
    compare(f"omol-l-2S+1={m}", [f"omol-{s}-2S+1={m}" for s in ("s", "m")])
