"""Plot the distributions binned by dataset_stats.py.

usage: python plot_dataset_stats.py STATS/histograms.npz [--split all] [-o distributions.png]

Only reads the pre-binned `histograms.npz` (a few hundred kB whatever the
dataset size), and draws every panel as one artist (`stairs` for histograms,
`pcolormesh` for the 2D panel), so a figure takes about a second.

  --split all              everything, filled histograms (default)
  --split train val test   the splits overlaid as outlines, normalised to the
                           fraction of each split, so they compare despite
                           their sizes (at most three, see COLORS)
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np


# reference palette of the dataviz skill, light surface. The first three
# categorical slots are the ones that stay distinguishable (also under colour
# vision deficiencies) when all of them share a panel.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

LINEAR_PANELS = {"mass_density", "number_density", "energy", "energy_per_atom"}
INTEGER_PANELS = {"n_atoms", "n_pairs"}

PANELS = [
    # name, title, x label, y label (count unit), log x
    ("n_atoms", "Atoms per structure", "atoms", "structures", True),
    ("n_periodic", "Periodic directions", "periodic directions", "structures", False),
    ("volume", "Cell volume (3D periodic)", "volume [Å³]", "structures", True),
    ("mass_density", "Mass density (3D periodic)", "density [g/cm³]", "structures", False),
    ("number_density", "Number density (3D periodic)", "atoms / Å³", "structures", False),
    ("energy", "Energy", "energy [eV]", "structures", False),
    ("energy_per_atom", "Energy per atom", "energy [eV/atom]", "structures", False),
    ("max_force", "Largest force per structure", "|F| max [eV/Å]", "structures", True),
    ("force_per_atom", "Force norm per atom", "|F| [eV/Å]", "atoms", True),
    ("n_pairs", "Neighbour list size ({cutoff} Å, full)", "pairs per structure + 1", "structures", True),
    ("neighbours_per_atom", "Neighbours per atom ({cutoff} Å)", "neighbours", "atoms", False),
    ("size_vs_neighbours", "Neighbours per atom vs size", "atoms per structure", "mean neighbours per atom", True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("histograms")
    parser.add_argument("--split", nargs="+", default=["all"])
    parser.add_argument("-o", "--output", default=None, help="default: distributions-<splits>.png next to the input")
    parser.add_argument("--dpi", type=int, default=130)
    args = parser.parse_args()
    if len(args.split) > len(COLORS):
        parser.error(f"at most {len(COLORS)} splits can share a panel")

    start = time.perf_counter()
    data = np.load(args.histograms)
    available = list(data["splits"])
    for split in args.split:
        if split not in available:
            parser.error(f"no split '{split}' in {args.histograms}, available: {', '.join(available)}")
    output = Path(args.output or Path(args.histograms).with_name(f"distributions-{'-'.join(args.split)}.png"))
    cutoff = float(data["cutoff"])

    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "path.simplify": True,
    })

    overlay = len(args.split) > 1
    get = lambda split, name, field: data[f"{split}/{name}/{field}"]  # noqa: E731

    def mark_median(ax, value, color, row, split):
        ax.axvline(value, color=color if overlay else INK_2, linewidth=1, linestyle="--")
        label = f"{split} median {value:.3g}" if overlay else f"median {value:.3g}"
        ax.annotate(label, (value, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -12 - 11 * row), textcoords="offset points", color=INK_2, fontsize=8)

    fig, axes = plt.subplots(3, 4, figsize=(17, 11), layout="constrained")
    for ax, (name, title, xlabel, ylabel, log_x) in zip(axes.ravel(), PANELS):
        title = title.format(cutoff=cutoff)

        if name == "size_vs_neighbours":
            # a 2D panel has no room for overlays: it shows the first split
            split = args.split[0]
            grid = get(split, name, "counts")
            masked = np.ma.masked_equal(grid.T, 0)
            mesh = ax.pcolormesh(get(split, name, "edges"), get(split, name, "y_edges"), masked,
                                 cmap="Blues", norm=matplotlib.colors.LogNorm(), rasterized=True)
            fig.colorbar(mesh, ax=ax, label="structures", pad=0.01)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(False)
            if overlay:
                title += f" ({split})"

        elif name == "n_periodic":
            width = 0.8 / len(args.split)
            for k, split in enumerate(args.split):
                counts = get(split, name, "counts").astype(float)
                shown = counts / counts.sum() if overlay else counts
                x = np.arange(4) + (k - (len(args.split) - 1) / 2) * width
                ax.bar(x, shown, width=width * 0.9, color=COLORS[k], label=split)
                if not overlay:
                    for d, count in enumerate(counts):
                        ax.annotate(f"{int(count):,}", (d, count), xytext=(0, 3), textcoords="offset points",
                                    ha="center", color=INK_2, fontsize=8)
            ax.set_xticks(range(4), ["0 (molecule)", "1", "2 (slab)", "3 (bulk)"])
            ax.grid(axis="x", visible=False)

        else:
            for k, split in enumerate(args.split):
                edges = get(split, name, "edges")
                counts = get(split, name, "counts").astype(float)
                if name == "force_per_atom":  # the first bin is [0, 1e-4): off a log axis
                    below, edges, counts = counts[0], edges[1:], counts[1:]
                if name in INTEGER_PANELS:  # per integer value, so wide bins do not stand out
                    counts = counts / np.maximum(np.diff(np.ceil(edges)), 1)
                if overlay:
                    counts = counts / get(split, name, "counts").sum()
                if name == "neighbours_per_atom":
                    last = np.nonzero(counts)[0].max() + 1
                    edges, counts = edges[: last + 1], counts[:last]
                if overlay:
                    ax.stairs(counts, edges, color=COLORS[k], linewidth=1.5, label=split)
                else:
                    ax.stairs(counts, edges, fill=True, color=COLORS[k])
                median = f"{split}/{name}/median"
                if median in data:
                    mark_median(ax, float(data[median]), COLORS[k], k, split)

            if name == "force_per_atom":
                nonzero = np.nonzero(counts)[0]
                ax.set_xlim(edges[nonzero[0]], edges[nonzero[-1] + 1])
                if not overlay:
                    xlabel += f"  ({int(below):,} atoms below 1e-4 not shown)"
            if name in LINEAR_PANELS:
                outside = max(float(get(s, name, "outside")) for s in args.split)
                xlabel += f"  ({100 * (1 - outside):.1f} % shown)"
            if name == "neighbours_per_atom":
                ax.set_xscale("symlog", linthresh=10)
                ax.set_xlim(left=-0.5)
            elif log_x:
                ax.set_xscale("log")
            ax.set_yscale("log")
            if name in INTEGER_PANELS:
                ylabel += " per value"

        if overlay:
            ylabel = f"fraction of {ylabel.split()[0]}" + (" per value" if "per value" in ylabel else "")
            if name == "size_vs_neighbours":
                ylabel = "mean neighbours per atom"
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    if overlay:
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncols=len(labels), frameon=False)
    header = ", ".join(
        f"{split}: {int(data[f'{split}/n_structures']):,} structures, {int(data[f'{split}/n_atom_total']):,} atoms"
        for split in args.split
    )
    fig.suptitle(header, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.savefig(output, dpi=args.dpi)
    print(f"wrote {output} in {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
