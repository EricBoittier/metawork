"""Flat-bottomed restraints that keep the ML shell waters near the peptide.

shell_restraints.py JOB_DIR [MARGIN_NM] [K]

For every water in the ML group (ml.ndx), adds a pull coordinate on the
distance between the peptide COM and the water COM, of type flat-bottom: zero
up to the water's distance in nvt.gro plus MARGIN_NM (default 0.10 nm), then
harmonic with force constant K (default 1000 kJ/mol/nm^2) beyond it. Inside
its sphere a water moves freely; it cannot leave. The potential depends on
positions only, so NVE still conserves the total energy (the pull energy is
reported as "COM Pull En.").

Appends one index group per water to index.ndx and writes pull.mdp, which
run_job.sh appends to the npt and nve parameters.
"""

import sys
from pathlib import Path

import numpy as np

job = Path(sys.argv[1])
margin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
k = float(sys.argv[3]) if len(sys.argv) > 3 else 1000.0
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06}

lines = (job / "nvt.gro").read_text().splitlines()
n = int(lines[1])
atoms = lines[2:2 + n]
resn = np.array([l[5:10].strip() for l in atoms])
name = np.array([l[10:15].strip() for l in atoms])
xyz = np.array([[float(l[20 + 8 * c:28 + 8 * c]) for c in range(3)] for l in atoms])
box = np.array([float(v) for v in lines[2 + n].split()[:3]])
mass = np.array([MASS["H" if a.startswith("H") else a[0]] for a in name])
mic = lambda d: d - box * np.round(d / box)

pep = np.where(resn != "SOL")[0]
p = xyz[pep[0]] + mic(xyz[pep] - xyz[pep[0]])
com_pep = (mass[pep, None] * p).sum(0) / mass[pep].sum()

ml = np.array([int(t) - 1 for l in (job / "ml.ndx").read_text().splitlines()[1:] for t in l.split()])
ow = ml[name[ml] == "OW"]
assert np.all(name[ow + 1] == "HW1") and np.all(name[ow + 2] == "HW2"), "unexpected water layout"

groups, pull, r0s = [], [], []
pull += ["", "; flat-bottomed restraints on the ML shell waters (shell_restraints.py)",
         "pull                     = yes",
         f"pull-ngroups             = {len(ow) + 1}",
         f"pull-ncoords             = {len(ow)}",
         "pull-group1-name         = Protein",
         "pull-nstxout             = 200",
         "pull-nstfout             = 0"]
for i, o in enumerate(ow, start=1):
    w = np.array([o, o + 1, o + 2])
    x = xyz[o] + mic(xyz[w] - xyz[o])
    com_w = (mass[w, None] * x).sum(0) / mass[w].sum()
    r0 = float(np.linalg.norm(mic(com_w - com_pep))) + margin
    r0s.append(r0)
    groups.append(f"[ shw{i} ]\n{o + 1} {o + 2} {o + 3}\n")
    pull += [f"pull-group{i + 1}-name        = shw{i}",
             f"pull-coord{i}-groups       = 1 {i + 1}",
             f"pull-coord{i}-type         = flat-bottom",
             f"pull-coord{i}-geometry     = distance",
             f"pull-coord{i}-dim          = Y Y Y",
             f"pull-coord{i}-start        = no",
             f"pull-coord{i}-init         = {r0:.4f}",
             f"pull-coord{i}-k            = {k:g}"]

assert max(r0s) < box.min() / 2, "restraint radius reaches half the box"
with open(job / "index.ndx", "a") as f:
    f.write("".join(groups))
(job / "pull.mdp").write_text("\n".join(pull) + "\n")
print(f"{len(ow)} shell waters restrained; radius {min(r0s):.3f}-{max(r0s):.3f} nm "
      f"(start + {margin} nm), k {k:g} kJ/mol/nm^2")
