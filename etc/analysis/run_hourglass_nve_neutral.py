"""
Runs the NVE engine-parity box-size scan for notebook 20, on `ch3f_neutral.xyz` --
the 5-atom, formally **neutral** fluoromethane fragment obtained by deleting the I-
counter-ion from notebook 17's 6-atom, net -1 `sn2_frame0.xyz` reference structure
(the other 5 atoms' positions are otherwise untouched).

Same two-part flow, and same two models (`sn2-matched-lorem`, `sn2-matched-pet`), as
`run_hourglass_nve.py` (notebook 17) -- see that script's docstring for the general
method. The point of this script is narrower: notebook 17 ended by conjecturing that
LOREM's periodic-vs-non-periodic NVE disagreement was a *physical* consequence of the
reference structure's net charge (periodic images of a net monopole interacting
across the box), and that a genuinely neutral test structure should make that
disagreement shrink much faster with box size (dipole-dipole images decay as
~1/L^3, not charge images' ~1/L). Running the exact same protocol on a neutral
structure is the direct test of that conjecture -- see notebook 20 for the result
(it does *not* hold up the way notebook 17 expected).

`hourglass_engines.run_ipi` needed one change to run a 5-atom structure instead of
the original 6-atom one: its i-PI driver template (`TEMPLATE_XYZ`) is now an
optional `template_xyz` argument (default unchanged) instead of a hardcoded global,
so this script can pass `ch3f_neutral.xyz` without touching notebook 17's own calls.

Produces:
  - results/hourglass_nve_neutral_boxscan.json (max|delta U| vs. box size)
  - figs/hourglass_nve_neutral_boxscan.png
  - results/hourglass_nve_neutral.json (headline ASE-vs-i-PI comparison at DEFAULT_CELL)
  - figs/hourglass_nve_neutral.png

Run with the standard torch env:
    /home/boittier/metawork/.venv/bin/python run_hourglass_nve_neutral.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sphericart.torch  # pre-import before loading any model
import ase.io

from hourglass_engines import run_ase, run_ipi, DEFAULT_CELL

OUTDIR = Path(__file__).parent
FIGDIR = OUTDIR / "figs"
RESULTS = OUTDIR / "results"
FIGDIR.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

STRUCTURE_XYZ = "ch3f_neutral.xyz"
MODELS = {
    "sn2-matched-lorem": "../data/sn2-matched-lorem/model.pt",
    "sn2-matched-pet": "../data/sn2-matched-pet/model.pt",
}

TIME_TOLERANCE = 1e-6  # fs
ENERGY_TOLERANCE = 0.001  # eV, same bound as the upstream cookbook recipe / notebook 17
BOX_SIZES = [20.0, 30.0, 50.0, 75.0, 100.0, 150.0]

atoms0 = ase.io.read(STRUCTURE_XYZ)
assert atoms0.get_chemical_symbols() == ["C", "H", "H", "H", "F"]

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})
ENGINE_COLORS = {"ASE": "#0072B2", "i-PI": "#D55E00"}
MODEL_COLORS = {"sn2-matched-lorem": "#D55E00", "sn2-matched-pet": "#009E73"}

# --- part 1: box-size scan (i-PI only, vs. the ASE reference) -----------

ase_trajectories = {}
default_box_ipi = {}
boxscan = {model_name: [] for model_name in MODELS}

for model_name, model_path in MODELS.items():
    print(f"Running ASE with {model_name} (NVE)")
    ase_times, ase_energies = run_ase(model_path, atoms0, ensemble="nve")
    ase_trajectories[model_name] = (ase_times, ase_energies)
    reference_energies = np.array(ase_energies)

    for box in BOX_SIZES:
        print(f"Running i-PI with {model_name} (NVE, box={box:g} Å)")
        ipi_times, ipi_energies = run_ipi(
            model_path, atoms0, ensemble="nve", cell=box, template_xyz=STRUCTURE_XYZ
        )
        max_delta = float(np.abs(np.array(ipi_energies) - reference_energies).max())
        boxscan[model_name].append({"box_size_angstrom": box, "max_abs_delta_U_meV": 1000 * max_delta})
        print(f"  max |ΔU| vs. ASE = {1000 * max_delta:.4f} meV")
        if box == DEFAULT_CELL:
            default_box_ipi[model_name] = (ipi_times, ipi_energies)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for model_name, rows in boxscan.items():
    boxes = [r["box_size_angstrom"] for r in rows]
    deltas = [r["max_abs_delta_U_meV"] for r in rows]
    ax.semilogy(boxes, deltas, "o-", color=MODEL_COLORS[model_name], linewidth=2, label=model_name)
ax.axhline(1000 * ENERGY_TOLERANCE, color="0.6", linestyle="--", linewidth=1,
           label=f"{1000 * ENERGY_TOLERANCE:.0f} meV tolerance")
ax.set_xlabel("i-PI vacuum cell edge (Å)")
ax.set_ylabel("max |ΔU| vs. ASE (meV, log scale)")
ax.set_title("Neutral CH3F: i-PI/ASE NVE agreement vs. box size")
ax.legend()
fig.tight_layout()
fig.savefig(FIGDIR / "hourglass_nve_neutral_boxscan.png")
plt.close(fig)

(RESULTS / "hourglass_nve_neutral_boxscan.json").write_text(json.dumps({
    "structure": STRUCTURE_XYZ,
    "box_sizes_angstrom": BOX_SIZES,
    "energy_tolerance_eV": ENERGY_TOLERANCE,
    "results": boxscan,
}, indent=2))
print("Wrote figs/hourglass_nve_neutral_boxscan.png and results/hourglass_nve_neutral_boxscan.json\n")

# --- part 2: ASE vs. i-PI at the chosen box size (DEFAULT_CELL) ---------

assert set(default_box_ipi) == set(MODELS), (
    f"DEFAULT_CELL={DEFAULT_CELL:g} Å must be one of BOX_SIZES so part 1's own run at that "
    f"size can be reused here instead of repeating it"
)

trajectories = {
    model_name: {"ASE": ase_trajectories[model_name], "i-PI": default_box_ipi[model_name]}
    for model_name in MODELS
}
comparison_rows = []

fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 4.2), constrained_layout=True, sharex=True)

for col, (model_name, engines) in enumerate(trajectories.items()):
    ax = axes[col]
    reference_times = np.array(engines["ASE"][0])
    reference_energies = np.array(engines["ASE"][1])
    for engine_name, (times, energies) in engines.items():
        times = np.array(times)
        energies = np.array(energies)
        ax.plot(times, energies, label=engine_name, color=ENGINE_COLORS[engine_name],
                linewidth=2, alpha=0.85)

        same_times = bool(len(times) == len(reference_times) and
                          np.abs(times - reference_times).max() < TIME_TOLERANCE)
        energy_error = float(np.abs(energies - reference_energies).max()) if same_times else None
        within_tolerance = bool(same_times and energy_error < ENERGY_TOLERANCE)
        comparison_rows.append({
            "model": model_name,
            "engine": engine_name,
            "same_times": same_times,
            "max_abs_delta_U_meV": 1000 * energy_error if energy_error is not None else None,
            "within_tolerance": within_tolerance,
        })
        status = "OK" if within_tolerance else "MISMATCH"
        print(f"{model_name} / {engine_name}: max |ΔU| = "
              f"{1000 * energy_error:.4f} meV [{status}]" if energy_error is not None else
              f"{model_name} / {engine_name}: time grids differ [MISMATCH]")
    ax.set_title(model_name)
    ax.set_xlabel("t / fs")
    ax.set_ylabel("U / eV")
    ax.legend()

fig.savefig(FIGDIR / "hourglass_nve_neutral.png")
plt.close(fig)

payload = {
    "structure": STRUCTURE_XYZ,
    "box_size_angstrom": DEFAULT_CELL,
    "models": list(MODELS.keys()),
    "engines": ["ASE", "i-PI"],
    "trajectories": {
        model_name: {
            engine_name: {"time_fs": times, "energy_eV": energies}
            for engine_name, (times, energies) in engines.items()
        }
        for model_name, engines in trajectories.items()
    },
    "comparison": comparison_rows,
    "time_tolerance_fs": TIME_TOLERANCE,
    "energy_tolerance_eV": ENERGY_TOLERANCE,
}
(RESULTS / "hourglass_nve_neutral.json").write_text(json.dumps(payload, indent=2))
print("Wrote figs/hourglass_nve_neutral.png and results/hourglass_nve_neutral.json")
