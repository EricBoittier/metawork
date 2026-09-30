"""PET-OMol-s: four heme sites as one system (charge -8, 2S+1 = 17) vs four systems (-2, 5)."""
import contextlib, io, sys
import numpy as np
from metatomic.torch.ase_calculator import MetatomicCalculator

EV_A = 964.853
src = open("site_test.py").read()
head = src[: src.index("forces = {}")]
clusters = []
for s in range(4):
    g = {"__name__": "site_test"}
    sys.argv = ["site_test.py", "../hb.gro", str(s)]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(head, g)
    clusters.append(g["cluster"])

calc = MetatomicCalculator("pet-omol-s-v1.0.0.pt", device="cuda")
separate = []
for c in clusters:
    a = c.copy()
    a.info.update(charge=-2, spin_multiplicity=5)
    a.calc = calc
    separate.append(a.get_forces())
combined = clusters[0].copy()
for c in clusters[1:]:
    combined += c
combined.info.update(charge=-8, spin_multiplicity=17)
combined.calc = calc
d = np.linalg.norm(combined.get_forces() - np.vstack(separate), axis=1) * EV_A
fe = [i for i, e in enumerate(combined.get_chemical_symbols()) if e == "Fe"]
p = combined.get_positions()
n = len(clusters[0])
gap = min(np.linalg.norm(p[i * n:(i + 1) * n, None] - p[None, j * n:(j + 1) * n], axis=2).min()
          for i in range(4) for j in range(i + 1, 4)) / 10
print(f"sites 4, atoms {len(combined)}, closest atoms of different sites {gap:.2f} nm")
print(f"combined vs separate: RMS {np.sqrt((d**2).mean()):.0f}, Fe " + " ".join(f"{d[i]:.0f}" for i in fe)
      + f", max {d.max():.0f} kJ/mol/nm")
