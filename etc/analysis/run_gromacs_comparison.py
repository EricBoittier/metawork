"""
Adds GROMACS to notebook 17's NVE engine comparison, now that `metawork/gromacs`
(branch `metatomic`, `GMX_METATOMIC=TORCH`) is built. Two things this
establishes, mirroring `run_lammps_comparison.py`:

1. GROMACS matches ASE/i-PI/LAMMPS well for the short-range PET model.
2. GROMACS is impractically slow for LOREM: a single 100-step run didn't
   finish in 10+ minutes even at the smallest box tried (20 Å), worse than
   either i-PI's or LAMMPS's own box-size walls for the same model. So unlike
   `run_lammps_comparison.py`, this script only exercises PET -- LOREM+GROMACS
   is a documented limitation (`hourglass_engines.py`'s docstring), not a
   routine comparison point.
3. Speed, repeated 3x on the same PET NVE trajectory, across all four engines
   now available (ASE, i-PI, LAMMPS, GROMACS) -- GROMACS spawns three
   subprocesses per call (`grompp`, `mdrun`, `gmx energy`), so its overhead is
   expected to dominate even for a model this cheap to evaluate.

Run with the standard torch env:
    /home/boittier/metawork/.venv/bin/python run_gromacs_comparison.py
"""
import json
import time

import numpy as np
import sphericart.torch  # pre-import before loading any model
import ase.io

from hourglass_engines import run_ase, run_ipi, run_lammps, run_gromacs
from model_registry import RESULTS

RESULTS.mkdir(exist_ok=True)

BOX = 50.0  # same practical box used for the LAMMPS three-engine comparison
PET_PATH = "../data/sn2-matched-pet/model.pt"
atoms0 = ase.io.read("sn2_frame0.xyz")

# --- part 1: GROMACS vs. ASE parity for PET -----------------------------

ase_t, ase_e = run_ase(PET_PATH, atoms0, ensemble="nve")
gmx_t, gmx_e = run_gromacs(PET_PATH, atoms0, ensemble="nve", cell=BOX)
gmx_delta_meV = 1000 * np.abs(np.array(gmx_e) - np.array(ase_e)).max()
print(f"GROMACS vs ASE (PET): max|dU| = {gmx_delta_meV:.4f} meV")

(RESULTS / "hourglass_nve_gromacs.json").write_text(json.dumps({
    "box_size_angstrom": BOX,
    "model": "sn2-matched-pet",
    "trajectories": {
        "ASE": {"time_fs": ase_t, "energy_eV": ase_e},
        "GROMACS": {"time_fs": gmx_t, "energy_eV": gmx_e},
    },
    "max_abs_delta_U_meV": gmx_delta_meV,
}, indent=2))
print("Wrote results/hourglass_nve_gromacs.json")

# --- part 2: four-engine speed comparison, PET, 3 trials ----------------

engines = {"ASE": run_ase, "i-PI": run_ipi, "LAMMPS": run_lammps, "GROMACS": run_gromacs}
speed_rows = []
for trial in range(3):
    row = {"trial": trial}
    for engine_name, run_engine in engines.items():
        kwargs = {} if engine_name == "ASE" else {"cell": BOX}
        t0 = time.perf_counter()
        run_engine(PET_PATH, atoms0, ensemble="nve", **kwargs)
        row[engine_name] = time.perf_counter() - t0
    speed_rows.append(row)
    print(f"trial {trial}: " + "  ".join(f"{k}={v:.2f}s" for k, v in row.items() if k != "trial"), flush=True)

(RESULTS / "hourglass_speed_comparison.json").write_text(json.dumps(speed_rows, indent=2))
print("Wrote results/hourglass_speed_comparison.json (now with GROMACS)")
