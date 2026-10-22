"""Model memory and time with the whole hemoglobin system (water and ions too) as ML (local helper).

fit_all.py MODEL [INFO_JSON] GRO: energy + forces for the atoms nearest heme 1's iron, from the
solute's size (9,026) out through the water, then all atoms in the periodic box; peak GPU memory
of the model alone (torch allocator) and time of the second call; stops at the first OOM.
"""
import json
import sys
import time

import ase
import numpy as np
import torch
from metatomic.torch.ase_calculator import MetatomicCalculator

model, gro = sys.argv[1], sys.argv[-1]
info = json.loads(sys.argv[2]) if len(sys.argv) > 3 else {}
lines = open(gro).read().splitlines()
n = int(lines[1])
at = lines[2 : 2 + n]
name = np.array([l[10:15].strip() for l in at])
X = np.array([[float(l[20 + 8 * k : 28 + 8 * k]) for k in range(3)] for l in at]) * 10
b = [float(v) * 10 for v in lines[2 + n].split()] + [0.0] * 6
cell = np.array([[b[0], b[3], b[4]], [b[5], b[1], b[6]], [b[7], b[8], b[2]]])  # gro: v1x v2y v3z v1y v1z v2x v2z v3x v3y
el = lambda a: {"FE": "Fe", "NA": "Na", "CL": "Cl"}.get(a, a[0])
fe = np.where(name == "FE")[0][0]
order = np.argsort(np.linalg.norm(X - X[fe], axis=1))

calc = MetatomicCalculator(model, device="cuda")
for size in (9026, 15000, 20000, 30000, 40000, n):
    periodic = size == n
    idx = order[:size]
    a = ase.Atoms([el(x) for x in name[idx]], positions=X[idx], cell=cell if periodic else None, pbc=periodic)
    a.info.update(info)
    a.calc = calc
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        a.get_forces()
        torch.cuda.synchronize()
        t1 = time.time()
        a.positions += 1e-4  # second call, without first-call overheads
        a.get_forces()
        torch.cuda.synchronize()
        t2 = time.time()
    except (torch.OutOfMemoryError, RuntimeError) as e:  # TorchScript raises OOM as RuntimeError
        if "out of memory" not in str(e):
            raise
        print(f"{size:6d} atoms{' (periodic)' if periodic else ''}: out of GPU memory", flush=True)
        break
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"{size:6d} atoms{' (periodic)' if periodic else ''}: {t2 - t1:7.3f} s per energy+forces, "
          f"peak {peak:6.2f} GiB, {0.0005 * 86400 / (t2 - t1) / 1000:.4f} ns/day at 0.5 fs", flush=True)
