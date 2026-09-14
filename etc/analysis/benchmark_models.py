"""
Loads every model in `model_registry.MODELS` and benchmarks load time and
single-point inference latency (energy + forces) on the shared reference
structure (`data/sn2/sn2.xyz`, frame 0). Produces the data behind
`01_load_models.ipynb`'s benchmark section.

Only the metatrain/torch, `MetatomicCalculator`-loadable models are included
(model_registry.MODELS) -- the native lorem-jax checkpoints (`LOREM_JAX_MODELS`)
need a different interpreter (`.venv-lorem-jax`) and calculator API, and the
torch-ported lorem-jax model (`PORTED_TORCH_MODELS`) loads through a custom
weight-loading path, not a plain `model.pt` export; both are out of scope for
a uniform load/inference benchmark and are already covered by notebooks
13-15.

Run with the standard torch env:
    /home/boittier/metawork/.venv/bin/python benchmark_models.py
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

from model_registry import MODELS, RESULTS

OUTDIR = Path(__file__).parent
FIGDIR = OUTDIR / "figs"
FIGDIR.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

N_WARMUP = 3
N_TIMED = 20

atoms0 = ase.io.read("../data/sn2/sn2.xyz", index=0)

rows = []
for m in MODELS:
    model_path = m["dir"] / "model.pt"
    if not model_path.exists():
        print(f"skip {m['key']}: no model.pt")
        continue

    from metatomic_ase import MetatomicCalculator

    t0 = time.perf_counter()
    calc = MetatomicCalculator(str(model_path))
    load_s = time.perf_counter() - t0

    atoms = atoms0.copy()
    atoms.calc = calc

    for _ in range(N_WARMUP):
        atoms.calc.results.clear()
        atoms.get_potential_energy()
        atoms.get_forces()

    call_times_ms = []
    for _ in range(N_TIMED):
        atoms.calc.results.clear()
        t0 = time.perf_counter()
        atoms.get_potential_energy()
        atoms.get_forces()
        call_times_ms.append(1000 * (time.perf_counter() - t0))

    row = {
        "key": m["key"],
        "title": m["title"],
        "family": m["family"],
        "dipole": m["dipole"],
        "load_time_s": load_s,
        "mean_call_ms": float(np.mean(call_times_ms)),
        "std_call_ms": float(np.std(call_times_ms)),
        "n_timed": N_TIMED,
    }
    rows.append(row)
    print(f"{m['key']:28s} load={load_s:6.3f}s  "
          f"call={row['mean_call_ms']:7.3f}+/-{row['std_call_ms']:.3f} ms")

(RESULTS / "benchmark_models.json").write_text(json.dumps(rows, indent=2))
print(f"\nWrote results/benchmark_models.json ({len(rows)} models)")

# --- plot: per-call inference latency, grouped by family, LOREM highlighted ---

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
FAMILY_COLORS = {"BPNN": "#0072B2", "PET": "#009E73", "LOREM": "#D55E00"}

order = sorted(range(len(rows)), key=lambda i: rows[i]["mean_call_ms"])
rows_sorted = [rows[i] for i in order]

fig, ax = plt.subplots(figsize=(8, 5.5))
y = np.arange(len(rows_sorted))
colors = [FAMILY_COLORS[r["family"]] for r in rows_sorted]
hatches = ["//" if r["dipole"] else None for r in rows_sorted]
bars = ax.barh(y, [r["mean_call_ms"] for r in rows_sorted],
                xerr=[r["std_call_ms"] for r in rows_sorted],
                color=colors, edgecolor="white", height=0.7,
                error_kw={"elinewidth": 1, "capsize": 2})
for bar, hatch, row in zip(bars, hatches, rows_sorted):
    if hatch:
        bar.set_hatch(hatch)
    if row["family"] == "LOREM":
        bar.set_linewidth(1.5)
        bar.set_edgecolor("black")

ax.set_yticks(y)
ax.set_yticklabels([r["key"] for r in rows_sorted])
ax.set_xlabel("mean energy+forces call latency (ms, warm)")
ax.set_title("Single-point inference cost -- SN2 model zoo")

from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()]
legend_elems.append(Patch(facecolor="0.6", hatch="//", label="+ dipole"))
ax.legend(handles=legend_elems, loc="lower right", fontsize=9)

fig.tight_layout()
fig.savefig(FIGDIR / "benchmark_models.png")
plt.close(fig)
print("Wrote figs/benchmark_models.png")
