"""
System-size scaling benchmark for the SN2 model zoo, in ASE (the one engine
besides i-PI actually usable in `metawork/.venv` -- see `hourglass_engines.py`'s
docstring; this is a different question from notebook 17's i-PI box-size
scan, which measured engine *parity* vs. box size, not raw compute cost vs.
system size).

Builds systems of N non-interacting copies of the reference SN2 unit
(`data/sn2/sn2.xyz`, frame 0, 6 atoms), spaced 30 A apart on a cubic grid so
short-range models never see cross-copy neighbors, and times energy+forces
calls as N grows. LOREM has no cutoff (`interaction_range = inf`, notebook
`01`) -- it evaluates a long-range term across every atom regardless of
spacing, so its cost is expected to grow faster with N than BPNN/PET's fixed-
cutoff neighbor-list evaluation, which stays ~O(N) once cross-copy neighbors
never form.

Stops early (adaptive) if a single call exceeds MAX_CALL_S, so this can't run
away the way the i-PI box scan did in notebook 17.

Run with the standard torch env:
    /home/boittier/metawork/.venv/bin/python benchmark_scaling.py
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sphericart.torch  # pre-import before loading any model
import ase.io
from ase import Atoms
from metatomic_ase import MetatomicCalculator

from model_registry import MODELS_BY_KEY, RESULTS

OUTDIR = Path(__file__).parent
FIGDIR = OUTDIR / "figs"
FIGDIR.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

SPACING = 30.0  # Å between copies -- well past every model's interaction range in notebook 12
N_COPIES = [1, 2, 4, 8, 16, 32]
N_WARMUP, N_TIMED = 4, 8
MAX_CALL_S = 8.0  # stop scaling this model up further once a single call exceeds this

MODEL_KEYS = ["sn2-matched-bpnn", "sn2-matched-pet", "sn2-matched-lorem"]

unit = ase.io.read("../data/sn2/sn2.xyz", index=0)


def make_system(n_copies, with_cell=False):
    """n_copies non-interacting replicas of `unit` on a cubic grid, SPACING apart.

    `with_cell=False` (default): no cell at all, matching how every other notebook
    in this series loads a non-periodic structure (`ase.io.read(...)` on a file with
    `pbc="F F F"` gives an empty cell) -- this isolates cost-vs-N with nothing else
    changing. `with_cell=True` reproduces this script's first (confounded) attempt,
    which defined an explicit large non-periodic cell and is kept only for the
    small side-by-side comparison below showing that choice's own fixed cost."""
    side = int(np.ceil(n_copies ** (1 / 3)))
    atoms = Atoms(cell=[side * SPACING] * 3, pbc=False) if with_cell else Atoms()
    count = 0
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if count >= n_copies:
                    break
                copy = unit.copy()
                copy.positions += np.array([i, j, k]) * SPACING
                atoms += copy
                count += 1
    return atoms


def time_one(model_path, atoms):
    atoms.calc = MetatomicCalculator(str(model_path))
    for _ in range(N_WARMUP):
        atoms.calc.results.clear()
        atoms.get_potential_energy(); atoms.get_forces()
    call_times = []
    for _ in range(N_TIMED):
        atoms.calc.results.clear()
        t0 = time.perf_counter()
        atoms.get_potential_energy(); atoms.get_forces()
        call_times.append(time.perf_counter() - t0)
    return float(np.mean(call_times)), float(np.std(call_times))


# --- main scan: cost vs. N, with no cell at all (isolates N as the only variable) ---

rows = []
for key in MODEL_KEYS:
    m = MODELS_BY_KEY[key]
    model_path = m["dir"] / "model.pt"

    for n in N_COPIES:
        atoms = make_system(n, with_cell=False)
        mean_s, std_s = time_one(model_path, atoms)
        rows.append({"key": key, "family": m["family"], "n_copies": n, "n_atoms": len(atoms),
                     "mean_call_s": mean_s, "std_call_s": std_s})
        print(f"{key:20s} n_atoms={len(atoms):4d}  call={mean_s * 1000:9.3f} ms", flush=True)

        if mean_s > MAX_CALL_S:
            print(f"  stopping {key}'s scan early: {mean_s:.1f}s > {MAX_CALL_S:.0f}s cap")
            break

(RESULTS / "benchmark_scaling.json").write_text(json.dumps(rows, indent=2))
print(f"\nWrote results/benchmark_scaling.json ({len(rows)} rows)")

# --- side comparison: does merely *defining* a large unused non-periodic cell ---
# --- cost anything by itself, independent of N? (one model, two sizes) ----------

cell_rows = []
probe_key = "sn2-matched-pet"
probe_path = MODELS_BY_KEY[probe_key]["dir"] / "model.pt"
for n in [min(N_COPIES), max(r["n_copies"] for r in rows if r["key"] == probe_key)]:
    for with_cell in [False, True]:
        atoms = make_system(n, with_cell=with_cell)
        mean_s, std_s = time_one(probe_path, atoms)
        cell_rows.append({"n_atoms": len(atoms), "with_cell": with_cell,
                           "mean_call_s": mean_s, "std_call_s": std_s})
        print(f"{probe_key} n_atoms={len(atoms):4d} with_cell={with_cell!s:5s} "
              f"call={mean_s * 1000:9.3f} ms", flush=True)

(RESULTS / "benchmark_scaling_cell_effect.json").write_text(json.dumps(cell_rows, indent=2))
print("\nWrote results/benchmark_scaling_cell_effect.json")
