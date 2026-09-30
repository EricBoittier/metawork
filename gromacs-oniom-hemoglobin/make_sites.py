"""Append the ML site groups to index.ndx: SITE1-4 (each heme with the side chain of its proximal
histidine, split from the ML group by nearest iron) and SOLUTE (protein + hemes, for the
all-protein ML test). Reads npt_whole.gro (npt.gro with whole molecules) and ml.ndx.
"""
import math
from pathlib import Path

here = Path(__file__).resolve().parent
lines = (here / "npt_whole.gro").read_text().splitlines()
n = int(lines[1])
atoms = lines[2 : 2 + n]
x = lambda i: [float(atoms[i][20 + 8 * k : 28 + 8 * k]) for k in range(3)]
name = lambda i: atoms[i][10:15].strip()
ml = [int(t) - 1 for t in (here / "ml.ndx").read_text().split("]", 1)[1].split()]
irons = [i for i in ml if name(i) == "FE"]
sites = {fe: [] for fe in irons}
for i in ml:
    sites[min(irons, key=lambda fe: math.dist(x(fe), x(i)))].append(i)
solute = [i for i in range(n) if atoms[i][5:10].strip() not in ("SOL", "NA", "CL")]


def group(title, idx):
    rows = (" ".join(str(a + 1) for a in idx[j : j + 15]) for j in range(0, len(idx), 15))
    return f"[ {title} ]\n" + "\n".join(rows) + "\n"


text = (here / "index.ndx").read_text()
if "[ SITE1 ]" not in text:
    text += "".join(group(f"SITE{k}", sites[fe]) for k, fe in enumerate(irons, 1)) + group("SOLUTE", solute)
    (here / "index.ndx").write_text(text)
print(", ".join(f"SITE{k}: {len(sites[fe])} atoms" for k, fe in enumerate(irons, 1)), f"; SOLUTE: {len(solute)} atoms")
