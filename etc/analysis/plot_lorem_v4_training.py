"""Plots old (version-3, drifted-checkpoint) vs. new (version-4, retrained)
sn2-matched-lorem training curves side by side, for notebook 21.

Run with the standard torch env:
    /home/boittier/metawork/.venv/bin/python plot_lorem_v4_training.py
"""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = Path(__file__).parent
FIGDIR = OUTDIR / "figs"
FIGDIR.mkdir(exist_ok=True)

OLD_CSV = "/home/boittier/data/sn2-matched-lorem/outputs/2026-09-07/09-41-35/train.csv"
NEW_CSV = "/home/boittier/data/sn2-matched-lorem-v4/outputs/2026-09-15/12-41-31/train.csv"

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


def load(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=2)
    return {
        "epoch": data[:, 0],
        "train_energy_rmse": data[:, 3],
        "train_forces_rmse": data[:, 4],
        "val_energy_rmse": data[:, 6],
        "val_forces_rmse": data[:, 7],
    }


old = load(OLD_CSV)
new = load(NEW_CSV)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
ax = axes[0]
ax.semilogy(old["epoch"], old["val_energy_rmse"], "o-", color="#D55E00", label="v3 (original, drifted checkpoint)")
ax.semilogy(new["epoch"], new["val_energy_rmse"], "o-", color="#009E73", label="v4 (retrained, this notebook)")
ax.set_xlabel("epoch")
ax.set_ylabel("validation energy RMSE (meV/atom, log scale)")
ax.set_title("Energy convergence")
ax.legend()

ax = axes[1]
ax.semilogy(old["epoch"], old["val_forces_rmse"], "o-", color="#D55E00", label="v3 (original)")
ax.semilogy(new["epoch"], new["val_forces_rmse"], "o-", color="#009E73", label="v4 (retrained)")
ax.set_xlabel("epoch")
ax.set_ylabel("validation forces RMSE (meV/Å, log scale)")
ax.set_title("Forces convergence")
ax.legend()

fig.suptitle("sn2-matched-lorem retrain: v3 vs. v4 architecture, same data/hypers/epochs")
fig.savefig(FIGDIR / "lorem_v4_training_curves.png")
plt.close(fig)
print("Wrote figs/lorem_v4_training_curves.png")
