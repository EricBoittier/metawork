"""
Runs the same NVE engine-parity protocol as `run_hourglass_nve_neutral.py`
(notebook 20) and `run_hourglass_nve.py` (notebook 17), but against the freshly
retrained `sn2-matched-lorem-v4/model.pt` -- the version-4-checkpoint replacement
for `sn2-matched-lorem` produced after fixing `LOREM.load_checkpoint`'s silent
`strict=False` parameter drop (see notebook 20 section 4 and
`metatrain` commit `9804dcbd`). Two questions this answers, both directly for
notebook 21:

1. Training analysis: does the retrain (same recipe, same data, current
   architecture) converge as cleanly as the original, and how do final
   accuracy/parameter counts compare? (Answered from `train_stdout.log`
   directly, not run here.)
2. Does the retrained model reproduce the *same* engine-parity picture as the
   old (broken-checkpoint-but-still-valid-`model.pt`) `sn2-matched-lorem`:
   the physically-expected ASE-vs-periodic gap on both the charged complex and
   the neutral CH3F fragment, with i-PI and LAMMPS still agreeing with *each
   other* (no new engine-specific bug introduced by retraining under the
   current architecture)?

Produces `results/lorem_v4_boxscan.json` and `results/lorem_v4_three_engine.json`.

Run with the standard torch env:
    /home/boittier/metawork/.venv/bin/python run_hourglass_nve_lorem_v4.py
"""
import json
from pathlib import Path

import numpy as np
import sphericart.torch  # pre-import before loading any model
import ase.io

from hourglass_engines import run_ase, run_ipi, run_lammps

OUTDIR = Path(__file__).parent
RESULTS = OUTDIR / "results"
RESULTS.mkdir(exist_ok=True)

MODEL_PATH = "/home/boittier/data/sn2-matched-lorem-v4/model.pt"
STRUCTURES = {
    "charged": "sn2_frame0.xyz",
    "neutral": "ch3f_neutral.xyz",
}
BOX_SIZES = [20.0, 30.0, 50.0, 75.0, 100.0, 150.0]
THREE_ENGINE_BOX = 50.0

# --- part 1: box-size scan, both structures -----------------------------

boxscan = {}
for structure_name, xyz in STRUCTURES.items():
    atoms0 = ase.io.read(xyz)
    print(f"[{structure_name}] Running ASE reference")
    _, ase_e = run_ase(MODEL_PATH, atoms0, ensemble="nve")
    ref = np.array(ase_e)
    rows = []
    for box in BOX_SIZES:
        print(f"[{structure_name}] Running i-PI, box={box:g} Å")
        _, ipi_e = run_ipi(MODEL_PATH, atoms0, ensemble="nve", cell=box, template_xyz=xyz)
        max_delta = float(np.abs(np.array(ipi_e) - ref).max())
        rows.append({"box_size_angstrom": box, "max_abs_delta_U_meV": 1000 * max_delta})
        print(f"  max |ΔU| vs. ASE = {1000 * max_delta:.4f} meV")
    boxscan[structure_name] = rows

(RESULTS / "lorem_v4_boxscan.json").write_text(json.dumps({
    "model_path": MODEL_PATH,
    "box_sizes_angstrom": BOX_SIZES,
    "results": boxscan,
}, indent=2))
print("Wrote results/lorem_v4_boxscan.json\n")

# --- part 2: i-PI vs. LAMMPS at a shared box, both structures ------------

three_engine = {}
for structure_name, xyz in STRUCTURES.items():
    atoms0 = ase.io.read(xyz)
    _, ase_e = run_ase(MODEL_PATH, atoms0, ensemble="nve")
    _, ipi_e = run_ipi(MODEL_PATH, atoms0, ensemble="nve", cell=THREE_ENGINE_BOX, template_xyz=xyz)
    _, lammps_e = run_lammps(MODEL_PATH, atoms0, ensemble="nve", cell=THREE_ENGINE_BOX)
    ase_e, ipi_e, lammps_e = np.array(ase_e), np.array(ipi_e), np.array(lammps_e)
    row = {
        "ipi_vs_ase_meV": 1000 * float(np.abs(ipi_e - ase_e).max()),
        "lammps_vs_ase_meV": 1000 * float(np.abs(lammps_e - ase_e).max()),
        "ipi_vs_lammps_meV": 1000 * float(np.abs(ipi_e - lammps_e).max()),
    }
    three_engine[structure_name] = row
    print(f"[{structure_name}] i-PI/ASE={row['ipi_vs_ase_meV']:.4f}  "
          f"LAMMPS/ASE={row['lammps_vs_ase_meV']:.4f}  "
          f"i-PI/LAMMPS={row['ipi_vs_lammps_meV']:.4f} meV")

(RESULTS / "lorem_v4_three_engine.json").write_text(json.dumps({
    "model_path": MODEL_PATH,
    "box_size_angstrom": THREE_ENGINE_BOX,
    "results": three_engine,
}, indent=2))
print("Wrote results/lorem_v4_three_engine.json")
