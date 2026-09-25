"""Plot the feature PCA: all of MAD-CORE as a density, one group on top in colour.

usage: python plot_pca.py PCA_DIR [--highlight OMol25] [--color n_atoms] [--pcs 1 2] [-o out.png]

Reads `projections.npy` and `pca.npz` from features_pca.py, the per-row
`dataset_group` labels from extract_groups.py (`--groups`), and, to colour the
highlighted rows, the per-structure statistics of dataset_stats.py
(`--stats`), mapped from (split, index in split) to global rows with the
fixed split (make_split.load_split).

Every row goes into a grey 2D histogram (log colour scale), so the 16.8M rows
cost one `pcolormesh`; the highlighted group is drawn as points, randomly
subsampled to `--max-points`, coloured by `--color`:

  n_atoms, n_pairs, max_force, volume (log scale); energy_per_atom,
  mass_density, number_density (linear); dataset (the group's dataset_id)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]  # the three all-pairs-safe slots
LOG_STATS = {"n_atoms", "n_pairs", "max_force", "volume"}
LABELS = {
    "n_atoms": "atoms per structure",
    "n_pairs": "neighbour pairs (4.5 Å)",
    "max_force": "largest force [eV/Å]",
    "volume": "cell volume [Å³]",
    "energy_per_atom": "energy per atom [eV]",
    "mass_density": "mass density [g/cm³]",
    "number_density": "atoms / Å³",
}


def per_row(stats_path, n_rows, name):
    """Statistic `name` of every global row (NaN where there is none)."""
    sys.path.insert(0, str(HERE))
    from make_split import load_split

    stats = np.load(stats_path, mmap_mode="r")
    covered = len(stats["split"])  # the stats cover the first `covered` rows
    rows = {k: v[v < covered] for k, v in load_split().items()}
    values = np.full(n_rows, np.nan)
    split, index = stats["split"][:], stats["index"][:]
    column = stats[name][:].astype(float)
    for code, split_name in enumerate(stats["splits"]):
        mask = split == code
        values[rows[str(split_name)][index[mask]]] = column[mask]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pca", type=Path)
    parser.add_argument("--groups", type=Path, default=Path("~/data/madcore/groups.npz").expanduser())
    parser.add_argument("--stats", type=Path, default=Path("~/data/madcore/stats-full/per-structure.npz").expanduser())
    parser.add_argument("--highlight", default="OMol25", help="dataset_group drawn in colour")
    parser.add_argument("--color", default="n_atoms", choices=[*LABELS, "dataset"])
    parser.add_argument("--pcs", type=int, nargs=2, default=[1, 2], help="1-based")
    parser.add_argument("--bins", type=int, default=400)
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    start = time.perf_counter()
    pca = np.load(args.pca / "pca.npz")
    projections = np.load(args.pca / "projections.npy", mmap_mode="r")
    a, b = (p - 1 for p in args.pcs)
    x, y = np.asarray(projections[:, a]), np.asarray(projections[:, b])
    present = np.isfinite(x) & np.isfinite(y)

    labels = np.load(args.groups)
    names = list(labels["groups"])
    if args.highlight not in names:
        parser.error(f"no dataset_group '{args.highlight}'")
    group = labels["group"][: len(x)]
    highlight = present & (group == names.index(args.highlight))

    rng = np.random.default_rng(0)
    shown = np.flatnonzero(highlight)
    if len(shown) > args.max_points:
        shown = np.sort(rng.choice(shown, args.max_points, replace=False))

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # the background covers the central 99.9 % of every row, in both directions
    lo_x, hi_x = np.percentile(x[present], [0.05, 99.95])
    lo_y, hi_y = np.percentile(y[present], [0.05, 99.95])
    pad_x, pad_y = 0.03 * (hi_x - lo_x), 0.03 * (hi_y - lo_y)
    x_edges = np.linspace(lo_x - pad_x, hi_x + pad_x, args.bins + 1)
    y_edges = np.linspace(lo_y - pad_y, hi_y + pad_y, args.bins + 1)
    density, _, _ = np.histogram2d(x[present], y[present], [x_edges, y_edges])

    fig, ax = plt.subplots(figsize=(8.5, 7), layout="constrained")
    greys = LinearSegmentedColormap.from_list("greys", ["#e9e8e4", "#3d3c39"])
    mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_equal(density.T, 0), cmap=greys,
                         norm=LogNorm(), rasterized=True)
    fig.colorbar(mesh, ax=ax, label="MAD-CORE structures per bin", pad=0.01, shrink=0.6, location="bottom", aspect=40)

    if args.color == "dataset":
        codes = labels["dataset"][: len(x)][shown]
        datasets = [str(labels["datasets"][c]) for c in np.unique(codes)]
        if len(datasets) > len(CATEGORICAL):
            parser.error(f"{args.highlight} has {len(datasets)} datasets, too many to colour apart")
        for k, (code, dataset) in enumerate(zip(np.unique(codes), datasets)):
            m = codes == code
            ax.scatter(x[shown][m], y[shown][m], s=3, color=CATEGORICAL[k], linewidths=0,
                       label=f"{dataset} ({m.sum():,})", rasterized=True)
        ax.legend(frameon=False, markerscale=4, loc="upper right")
    else:
        values = per_row(args.stats, len(x), args.color)[shown]
        ok = np.isfinite(values) & ((values > 0) if args.color in LOG_STATS else True)
        norm = LogNorm() if args.color in LOG_STATS else None
        if norm is None:
            lo, hi = np.percentile(values[ok], [1, 99])
            norm = matplotlib.colors.Normalize(lo, hi)
        points = ax.scatter(x[shown][ok], y[shown][ok], c=values[ok], s=3, cmap="viridis", norm=norm,
                            linewidths=0, rasterized=True)
        fig.colorbar(points, ax=ax, label=f"{args.highlight}: {LABELS[args.color]}", pad=0.01, shrink=0.8)
        if (~ok).any():
            print(f"{(~ok).sum():,} of the drawn {args.highlight} rows have no {args.color}")

    explained = pca["explained"]
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_xlabel(f"PC{args.pcs[0]} ({100 * explained[a]:.1f} % of the variance)")
    ax.set_ylabel(f"PC{args.pcs[1]} ({100 * explained[b]:.1f} % of the variance)")
    subsample = f", {len(shown):,} drawn" if len(shown) < highlight.sum() else ""
    ax.set_title(
        f"MAD-CORE feature PCA, fitted on all {present.sum():,} structures (grey)\n"
        f"{args.highlight}: {highlight.sum():,} structures{subsample} (colour)",
        loc="left",
    )

    output = args.output or args.pca / f"pca-pc{args.pcs[0]}{args.pcs[1]}-{args.highlight}-{args.color}.png"
    fig.savefig(output, dpi=args.dpi)
    print(f"wrote {output} in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
