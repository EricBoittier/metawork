"""
Adds LAMMPS and GROMACS to notebook 20's neutral-CH3F NVE comparison, mirroring
`run_lammps_comparison.py`/`run_gromacs_comparison.py` (notebook 17) at the same
shared THREE_ENGINE_BOX=50 Å. Two things checked here, both already true for the
charged reference structure and re-verified on the neutral one:

1. **i-PI and LAMMPS agree with each other almost exactly** for LOREM at box=50 Å,
   even though both disagree with the non-periodic ASE reference by ~360 meV
   (see notebook 20, part 1) -- independent confirmation (two separate Ewald
   implementations) that the ASE/periodic gap is a property of evaluating this
   model under *any* periodicity, not one engine's own quirk.
2. **GROMACS remains impractical for LOREM** even on this much smaller (5-atom,
   neutral) structure: a single 100-step run didn't finish in 5 minutes (see
   notebook 20's investigation log), worse than i-PI's or LAMMPS's own box-size
   walls. So, like `run_gromacs_comparison.py`, only PET is exercised here.

Run with the standard torch env:
    /home/boittier/metawork/.venv/bin/python run_hourglass_nve_neutral_multiengine.py
"""
import json

import numpy as np
import sphericart.torch  # pre-import before loading any model
import ase.io

from hourglass_engines import run_ase, run_ipi, run_lammps, run_gromacs
from model_registry import RESULTS

RESULTS.mkdir(exist_ok=True)

STRUCTURE_XYZ = "ch3f_neutral.xyz"
BOX = 50.0
MODELS = {
    "sn2-matched-lorem": "../data/sn2-matched-lorem/model.pt",
    "sn2-matched-pet": "../data/sn2-matched-pet/model.pt",
}

atoms0 = ase.io.read(STRUCTURE_XYZ)

# --- part 1: ASE / i-PI / LAMMPS, both models ---------------------------

trajectories = {}
for model_name, model_path in MODELS.items():
    ase_t, ase_e = run_ase(model_path, atoms0, ensemble="nve")
    ipi_t, ipi_e = run_ipi(model_path, atoms0, ensemble="nve", cell=BOX, template_xyz=STRUCTURE_XYZ)
    lammps_t, lammps_e = run_lammps(model_path, atoms0, ensemble="nve", cell=BOX)
    trajectories[model_name] = {"ASE": (ase_t, ase_e), "i-PI": (ipi_t, ipi_e), "LAMMPS": (lammps_t, lammps_e)}
    print(f"{model_name} done", flush=True)

comparison_rows = []
for model_name, engines in trajectories.items():
    ref_e = np.array(engines["ASE"][1])
    for engine_name, (t, e) in engines.items():
        delta_meV = 1000 * np.abs(np.array(e) - ref_e).max()
        comparison_rows.append({"model": model_name, "engine": engine_name, "max_abs_delta_U_meV": delta_meV})
        print(f"{model_name:20s} {engine_name:8s} max|dU| vs ASE = {delta_meV:.4f} meV")

ipi_lammps_lorem = 1000 * np.abs(
    np.array(trajectories["sn2-matched-lorem"]["LAMMPS"][1])
    - np.array(trajectories["sn2-matched-lorem"]["i-PI"][1])
).max()
print(f"\nLOREM, i-PI vs LAMMPS directly (both periodic, box={BOX:g} Å): {ipi_lammps_lorem:.4f} meV")

(RESULTS / "hourglass_nve_neutral_lammps.json").write_text(json.dumps({
    "structure": STRUCTURE_XYZ,
    "box_size_angstrom": BOX,
    "trajectories": {
        m: {eng: {"time_fs": t, "energy_eV": e} for eng, (t, e) in engs.items()}
        for m, engs in trajectories.items()
    },
    "comparison": comparison_rows,
    "lorem_ipi_vs_lammps_meV": ipi_lammps_lorem,
}, indent=2))
print("\nWrote results/hourglass_nve_neutral_lammps.json")

# --- part 2: GROMACS, PET only (LOREM+GROMACS remains impractical) ------

pet_path = MODELS["sn2-matched-pet"]
gmx_t, gmx_e = run_gromacs(pet_path, atoms0, ensemble="nve", cell=BOX)
gmx_delta_meV = 1000 * np.abs(np.array(gmx_e) - np.array(trajectories["sn2-matched-pet"]["ASE"][1])).max()
print(f"GROMACS vs ASE (PET, neutral CH3F): max|dU| = {gmx_delta_meV:.4f} meV")

(RESULTS / "hourglass_nve_neutral_gromacs.json").write_text(json.dumps({
    "structure": STRUCTURE_XYZ,
    "box_size_angstrom": BOX,
    "model": "sn2-matched-pet",
    "trajectories": {
        "ASE": {"time_fs": trajectories["sn2-matched-pet"]["ASE"][0], "energy_eV": trajectories["sn2-matched-pet"]["ASE"][1]},
        "GROMACS": {"time_fs": gmx_t, "energy_eV": gmx_e},
    },
    "max_abs_delta_U_meV": gmx_delta_meV,
}, indent=2))
print("Wrote results/hourglass_nve_neutral_gromacs.json")
