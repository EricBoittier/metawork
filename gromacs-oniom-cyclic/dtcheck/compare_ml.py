import sys, numpy as np, ase
from metatomic.torch.ase_calculator import MetatomicCalculator
gro, xvg = sys.argv[1], sys.argv[2]
lines = open(gro).read().splitlines()
n = int(lines[1]); step = n + 3
frames = [lines[i:i + step] for i in range(0, len(lines), step)]
times = [float(f[0].split("t=")[1].split()[0]) for f in frames]
mk = lambda f: ase.Atoms([l[10:15].strip()[0] for l in f[2:2 + n]],
                         positions=[[float(l[20 + 8 * k:28 + 8 * k]) * 10 for k in range(3)] for l in f[2:2 + n]])
calc = MetatomicCalculator("pet-mad-xs-v1.5.0.pt", device="cuda")
eV = 96.48533212
ref = []
for f in frames:
    a = mk(f); a.calc = calc; ref.append(a.get_potential_energy() * eV)
ref = np.array(ref)
x = np.loadtxt(xvg, comments=["#", "@"])
gmx = np.interp(times, x[:, 0], x[:, 1])
d = gmx - ref
print(mk(frames[0]).get_chemical_formula(), len(frames), "frames")
print("GROMACS - python (kJ/mol): mean %.2f std %.2f min %.2f max %.2f" % (d.mean(), d.std(), d.min(), d.max()))
print(np.round(d[:10], 2).tolist())
