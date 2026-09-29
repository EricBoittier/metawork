import numpy as np, ase, torch
from metatomic.torch.ase_calculator import MetatomicCalculator
H, atoms, n_pep = 0.002, [0, 1, 9, 20, 40, 60, 86], 87
e = np.loadtxt("mlpot.xvg", comments=["#", "@"]); ML, POT = e[:, 1], e[:, 2]
lines = open("frames.gro").read().splitlines(); n = int(lines[1])
pep = lines[2:2 + n_pep]
a = ase.Atoms([l[10:15].strip()[0] for l in pep], positions=[[float(l[20 + 8*k:28 + 8*k]) * 10 for k in range(3)] for l in pep])
# the peptide is whole in nve.gro? check max bond-ish extent
print("peptide extent (A)", np.ptp(a.positions, axis=0).round(1))
a.calc = MetatomicCalculator("pet-mad-xs-v1.5.0.pt", device="cuda")
Fpy = a.get_forces() * 96.48533212 * 10  # eV/A -> kJ/mol/nm
dump = open("dump.txt").read().split("f (")[1].splitlines()[1:n + 1]
F = np.array([[float(v) for v in l.split("{")[1].split("}")[0].split(",")] for l in dump])
i = 1
for at in atoms:
    fdml, fdmm = [], []
    for k in range(3):
        fdml.append(-(ML[i] - ML[i+1]) / (2*H)); fdmm.append(-((POT-ML)[i] - (POT-ML)[i+1]) / (2*H)); i += 2
    fdml, fdmm = np.array(fdml), np.array(fdmm)
    print(f"atom {at+1:3d} ML: FD {fdml.round(0)} py {Fpy[at].round(0)} | MM: FD {fdmm.round(0)} gmx-py {(F[at]-Fpy[at]).round(0)}")
