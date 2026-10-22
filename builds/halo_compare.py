"""Compare -rerun results: Metatomic energy and all forces of each run against the first.

halo_compare.py GMX REF RUN...   (deffnm prefixes in the current directory)
"""
import subprocess
import sys

import numpy as np

gmx, ref, runs = sys.argv[1], sys.argv[2], sys.argv[3:]


def xvg(path):
    return np.loadtxt([l for l in open(path) if l[0] not in "#@"], ndmin=2)


def energy(tag, term="Metatomic"):
    """First value of an energy-file term, nan if the term is missing"""
    out = f"_{tag}_{term}.xvg"
    done = subprocess.run([gmx, "energy", "-f", f"{tag}.edr", "-o", out, "-quiet"], input=f"{term}\n0\n",
                          text=True, capture_output=True)
    return xvg(out)[:, 1] if done.returncode == 0 else np.array([np.nan])


def forces(tag):
    out = f"_{tag}_f.xvg"
    tpr = ref.rsplit("-", 1)[0] + ".tpr"  # MODEL-TAG -> MODEL.tpr
    subprocess.run([gmx, "traj", "-s", tpr, "-f", f"{tag}.trr", "-of", out, "-quiet"], input="0\n",
                   text=True, capture_output=True, check=True)
    return xvg(out)[:, 1:]


E0, F0 = energy(ref), forces(ref)
print(f"  {ref}: Metatomic energy {E0[0]:.4f} kJ/mol, |F| max {np.abs(F0).max():.1f} kJ/mol/nm")
for tag in runs:
    E, F = energy(tag), forces(tag)
    d = np.abs(F - F0)
    vir = "  ".join(f"{t} {energy(tag, t)[0]:.1f} vs {energy(ref, t)[0]:.1f}" for t in ("Vir-XX", "Vir-YY", "Vir-ZZ"))
    print(f"  {tag}: dE = {E[0] - E0[0]:+.4f} kJ/mol, max |dF| = {d.max():.4f}, rms dF = {np.sqrt((d ** 2).mean()):.5f} kJ/mol/nm")
    print(f"  {tag}: virial (kJ/mol) {vir}")
