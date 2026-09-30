"""Energy, temperature and Fe-NE2 time series of the hemoglobin ONIOM runs -> timeseries.json.

For each run (PET-MAD xs: onvt/onve, PET-OMol s: omvt/omve) reads the NVT and NVE energy
files and the four proximal His NE2 - Fe distances from the compressed trajectories.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent  # a run directory
G = os.environ["GMX"]  # from etc/oniom/env.sh
RUNS = {"pet-mad-xs": ("onvt", "onve"), "pet-omol-s": ("omvt", "omve")}


def xvg(path):
    lines = path.read_text().splitlines()
    names = [l.split('"')[1] for l in lines if l.startswith("@ s") and 'legend "' in l]
    data = np.loadtxt([l for l in lines if l and l[0] not in "#@"], ndmin=2)
    return data[:, 0], {nm: data[:, k + 1] for k, nm in enumerate(names)}


def energy(stem):
    out = HERE / f"_{stem}_e.xvg"
    subprocess.run([G, "energy", "-f", f"{stem}.edr", "-o", out.name], cwd=HERE, capture_output=True,
                   input="Total-Energy\nTemperature\nMetatomic-Potential\n0\n", text=True, check=True)
    t, e = xvg(out)
    out.unlink()
    return t, e


def fe_ne2(stem):
    groups, cur = {}, None
    for l in open(HERE / "index.ndx"):
        if l.startswith("["):
            cur = l.strip("[] \n")
            groups[cur] = []
        else:
            groups[cur] += [int(t) for t in l.split()]
    names = [l[10:15].strip() for l in (HERE / "npt.gro").read_text().splitlines()[2:-1]]
    pairs = []
    for k in range(1, 5):
        site = groups[f"SITE{k}"]
        fe = [a for a in site if names[a - 1] == "FE"][0]
        ne = [a for a in site if names[a - 1] == "NE2"][0]
        pairs.append(f"atomnr {fe} {ne}")
    out = HERE / f"_{stem}_d.xvg"
    subprocess.run([G, "distance", "-s", f"{stem}.tpr", "-f", f"{stem}.xtc", "-oall", out.name, "-select", *pairs],
                   cwd=HERE, capture_output=True, text=True, check=True)
    lines = [l for l in out.read_text().splitlines() if l and l[0] not in "#@"]
    out.unlink()
    d = np.loadtxt(lines, ndmin=2)
    return d[:, 0], d[:, 1:]


result = {}
for model, (nvt, nve) in RUNS.items():
    if not (HERE / f"{nve}.gro").exists():
        continue
    r = {}
    t1, e1 = energy(nvt)
    t2, e2 = energy(nve)
    tn, dn = fe_ne2(nvt)
    tv, dv = fe_ne2(nve)
    fit = np.polyfit(t2 - t2[0], e2["Total Energy"], 1)
    r["nvt"] = {"t": (t1 - t1[0]).round(3).tolist(), "T": e1["Temperature"].round(2).tolist(),
                "ml": (e1["Metatomic Potential"] - e1["Metatomic Potential"][0]).round(1).tolist()}
    step = max(1, len(t2) // 500)
    r["nve"] = {"t": (t2 - t2[0])[::step].round(3).tolist(),
                "etot": (e2["Total Energy"] - e2["Total Energy"][0])[::step].round(2).tolist(),
                "T": e2["Temperature"][::step].round(2).tolist(),
                "ml": (e2["Metatomic Potential"] - e2["Metatomic Potential"][0])[::step].round(1).tolist()}
    # NVE times restart; place its frames after the NVT ones
    r["fe_ne2"] = {"t": np.concatenate([tn - tn[0], tv - tv[0] + (tn[-1] - tn[0])]).round(3).tolist(),
                   "d": np.vstack([dn, dv]).T.round(4).tolist()}
    resid = e2["Total Energy"] - np.polyval(fit, t2 - t2[0])
    r["stats"] = {"drift": float(fit[0]), "std": float(resid.std()), "T": float(e2["Temperature"].mean()),
                  "fe_ne2_end": dv[-1].round(4).tolist(), "fe_ne2_mean_nve": dv.mean(0).round(4).tolist()}
    result[model] = r
(HERE / "timeseries.json").write_text(json.dumps(result, separators=(",", ":")))
for m, r in result.items():
    print(m, r["stats"])
