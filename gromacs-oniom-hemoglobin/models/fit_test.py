"""Does a model fit on the GPU with the whole protein as ML? fit_test.py MODEL [INFO_JSON]

Energy + forces for growing parts of hemoglobin (protein + hemes, no water), from the atoms
around heme 1 out to all 9026 atoms; prints time and peak GPU memory per size, and stops
at the first size that runs out of memory.
"""
import json
import sys
import time

import ase
import numpy as np
import torch
from metatomic.torch.ase_calculator import MetatomicCalculator

model = sys.argv[1]
info = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
lines = open("../npt_whole.gro").read().splitlines()
n = int(lines[1])
at = lines[2 : 2 + n]
resn = np.array([l[5:10].strip() for l in at])
name = np.array([l[10:15].strip() for l in at])
X = np.array([[float(l[20 + 8 * k : 28 + 8 * k]) for k in range(3)] for l in at]) * 10
solute = np.where(~np.isin(resn, ["SOL", "NA", "CL"]))[0]
fe = solute[name[solute] == "FE"][0]
order = solute[np.argsort(np.linalg.norm(X[solute] - X[fe], axis=1))]
el = lambda a: "Fe" if a == "FE" else a[0]

calc = MetatomicCalculator(model, device="cuda")
for size in (500, 1000, 2000, 4000, 6000, len(solute)):
    idx = order[:size]
    a = ase.Atoms([el(x) for x in name[idx]], positions=X[idx])
    a.info.update(info)
    a.calc = calc
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time()
        a.get_forces()
        torch.cuda.synchronize()
        t1 = time.time()
        a.positions += 1e-4  # second call, without first-call overheads
        a.get_forces()
        torch.cuda.synchronize()
        t2 = time.time()
    except torch.OutOfMemoryError:
        print(f"{size:6d} atoms: out of GPU memory", flush=True)
        break
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"{size:6d} atoms: {t2 - t1:7.3f} s per energy+forces, peak {peak:6.2f} GiB", flush=True)
