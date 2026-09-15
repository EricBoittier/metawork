"""Plots the v3-vs-v4, charged-vs-neutral box-scan comparison figure for notebook 21.

Run with the standard torch env:
    /home/boittier/metawork/.venv/bin/python plot_lorem_v4_boxscan.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = Path(__file__).parent
FIGDIR = OUTDIR / "figs"
RESULTS = OUTDIR / "results"
FIGDIR.mkdir(exist_ok=True)

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

v3_charged_boxscan = json.loads((RESULTS / "hourglass_nve_boxscan.json").read_text())
v3_neutral_boxscan = json.loads((RESULTS / "hourglass_nve_neutral_boxscan.json").read_text())
v4_boxscan = json.loads((RESULTS / "lorem_v4_boxscan.json").read_text())

v3_charged = {r["box_size_angstrom"]: r["max_abs_delta_U_meV"] for r in v3_charged_boxscan["results"]["sn2-matched-lorem"]}
v3_neutral = {r["box_size_angstrom"]: r["max_abs_delta_U_meV"] for r in v3_neutral_boxscan["results"]["sn2-matched-lorem"]}
v4_charged = {r["box_size_angstrom"]: r["max_abs_delta_U_meV"] for r in v4_boxscan["results"]["charged"]}
v4_neutral = {r["box_size_angstrom"]: r["max_abs_delta_U_meV"] for r in v4_boxscan["results"]["neutral"]}
box_sizes = v4_boxscan["box_sizes_angstrom"]

MODEL_COLORS = {"charged v3": "#D55E00", "charged v4": "#CC3311",
                 "neutral v3": "#0072B2", "neutral v4": "#009E73"}

fig, ax = plt.subplots(figsize=(7, 5))
for label, series in [
    ("charged v3", v3_charged), ("charged v4", v4_charged),
    ("neutral v3", v3_neutral), ("neutral v4", v4_neutral),
]:
    deltas = [series[b] for b in box_sizes]
    style = "o-" if "v4" in label else "s--"
    ax.semilogy(box_sizes, deltas, style, color=MODEL_COLORS[label], linewidth=2, label=label)
ax.axhline(1.0, color="0.6", linestyle=":", linewidth=1, label="1 meV tolerance")
ax.set_xlabel("i-PI vacuum cell edge (Å)")
ax.set_ylabel("max |ΔU| vs. ASE (meV, log scale)")
ax.set_title("sn2-matched-lorem: v3 vs. v4, charged vs. neutral")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIGDIR / "lorem_v4_boxscan_comparison.png")
plt.close(fig)
print("Wrote figs/lorem_v4_boxscan_comparison.png")
