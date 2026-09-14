"""
Adds LAMMPS to notebook 17's NVE engine-parity comparison, now that
`metawork/lammps` (branch `metatomic`, `PKG_ML-METATOMIC`) is built --
`hourglass_engines.LMP_BINARY`. Two things this script establishes:

1. LAMMPS's own practical box-size ceiling for LOREM is tighter than i-PI's:
   a single 100-step run at box=150 Å (notebook 17's `DEFAULT_CELL`, chosen
   for ASE/i-PI) didn't finish in 120s in an isolated/cold process, against
   ~23-26s at box=50 Å for both i-PI and LAMMPS. So the three-engine
   comparison below uses box=50 Å, not `DEFAULT_CELL` -- the common ground
   all three engines can actually run in practical time.
2. Wall-clock speed, repeated 3x on the same PET NVE trajectory to separate
   real steady-state cost from one-time JIT/compile warm-up (ASE and i-PI
   run in one persistent Python process and show a slower first call; LAMMPS
   spawns a fresh subprocess every single invocation, so it never gets to
   amortize a warm-up the way the other two do, and its timings are flat
   across trials as a result).

Run with the standard torch env:
    /home/boittier/metawork/.venv/bin/python run_lammps_comparison.py
"""
import json
import time
from pathlib import Path

import numpy as np
import sphericart.torch  # pre-import before loading any model
import ase.io

from hourglass_engines import run_ase, run_ipi, run_lammps
from model_registry import RESULTS

RESULTS.mkdir(exist_ok=True)

THREE_ENGINE_BOX = 50.0  # Å -- practical for ASE, i-PI, and LAMMPS alike (see docstring)
MODELS = {
    "sn2-matched-lorem": "../data/sn2-matched-lorem/model.pt",
    "sn2-matched-pet": "../data/sn2-matched-pet/model.pt",
}
ENGINES = {"ASE": run_ase, "i-PI": run_ipi, "LAMMPS": run_lammps}

atoms0 = ase.io.read("sn2_frame0.xyz")

# --- part 1: three-engine NVE trajectories + parity table, both models -----

trajectories = {}
for model_name, model_path in MODELS.items():
    trajectories[model_name] = {}
    for engine_name, run_engine in ENGINES.items():
        kwargs = {} if engine_name == "ASE" else {"cell": THREE_ENGINE_BOX}
        t, e = run_engine(model_path, atoms0, ensemble="nve", **kwargs)
        trajectories[model_name][engine_name] = (t, e)
        print(f"{model_name:20s} {engine_name:8s} done", flush=True)

comparison_rows = []
for model_name, engines in trajectories.items():
    ref_e = np.array(engines["ASE"][1])
    for engine_name, (t, e) in engines.items():
        delta_meV = 1000 * np.abs(np.array(e) - ref_e).max()
        comparison_rows.append({"model": model_name, "engine": engine_name, "max_abs_delta_U_meV": delta_meV})
        print(f"{model_name:20s} {engine_name:8s} max|dU| vs ASE = {delta_meV:.4f} meV")

(RESULTS / "hourglass_nve_lammps.json").write_text(json.dumps({
    "box_size_angstrom": THREE_ENGINE_BOX,
    "trajectories": {
        m: {eng: {"time_fs": t, "energy_eV": e} for eng, (t, e) in engs.items()}
        for m, engs in trajectories.items()
    },
    "comparison": comparison_rows,
}, indent=2))
print("\nWrote results/hourglass_nve_lammps.json")

# --- part 2: repeated wall-clock speed, PET, 3 trials per engine -----------

speed_path = "../data/sn2-matched-pet/model.pt"
speed_rows = []
for trial in range(3):
    row = {"trial": trial}
    for engine_name, run_engine in ENGINES.items():
        kwargs = {} if engine_name == "ASE" else {"cell": THREE_ENGINE_BOX}
        t0 = time.perf_counter()
        run_engine(speed_path, atoms0, ensemble="nve", **kwargs)
        row[engine_name] = time.perf_counter() - t0
    speed_rows.append(row)
    print(f"trial {trial}: " + "  ".join(f"{k}={v:.2f}s" for k, v in row.items() if k != "trial"), flush=True)

(RESULTS / "hourglass_speed_comparison.json").write_text(json.dumps(speed_rows, indent=2))
print("\nWrote results/hourglass_speed_comparison.json")
