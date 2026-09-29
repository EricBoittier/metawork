"""Central finite differences of the potential energy vs GROMACS forces, via mdrun -rerun."""
import subprocess, sys, numpy as np
G, tpr, gro, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
atoms = [int(a) for a in sys.argv[5].split(",")]  # 0-based
H = 0.002
lines = open(gro).read().splitlines()
n = int(lines[1])
xyz = np.array([[float(l[20 + 8 * k:28 + 8 * k]) for k in range(3)] for l in lines[2:2 + n]])
fmt = lambda l, x: l[:20] + "".join("%8.3f" % v for v in x) + l[44:]
frames = [xyz]
for a in atoms:
    for k in range(3):
        for s in (+1, -1):
            x = xyz.copy(); x[a, k] += s * H; frames.append(x)
with open(f"{out}/frames.gro", "w") as f:
    for i, x in enumerate(frames):
        f.write(f"frame t= {i}.0\n{n}\n" + "".join(fmt(l, x[j]) + "\n" for j, l in enumerate(lines[2:2 + n])) + lines[2 + n] + "\n")
run = lambda *a: subprocess.run([G, *a], cwd=out, capture_output=True, text=True, input="Potential\n0\n", check=True)
import os
if not os.path.exists(f"{out}/rr.edr"): run("mdrun", "-s", tpr, "-rerun", "frames.gro", "-deffnm", "rr", "-nt", "16")
run("energy", "-f", "rr.edr", "-o", "pot.xvg")
E = np.loadtxt(f"{out}/pot.xvg", comments=["#", "@"])[:, 1]
dump = subprocess.run([G, "dump", "-f", "rr.trr"], cwd=out, capture_output=True, text=True).stdout
fl = dump.split("f (")[1].splitlines()[1:n + 1]
F = np.array([[float(v) for v in l.split("{")[1].split("}")[0].split(",")] for l in fl])
i = 1
for a in atoms:
    fd = []
    for k in range(3):
        fd.append(-(E[i] - E[i + 1]) / (2 * H)); i += 2
    print(f"atom {a + 1:5d}: gmx F {np.round(F[a], 1)}  FD {np.round(fd, 1)}  |diff| {np.linalg.norm(F[a] - fd):.1f}")
