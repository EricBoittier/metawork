"""Distributions over every structure of MemmapDataset directories.

usage: python dataset_stats.py DIR... [-o OUT] [--cutoff 4.5] [--workers N]

Reads the directories written by `madcore-to-diskdataset` (e.g.
`~/data/madcore/memmap-2pow18/{train,val,test}`) through metatrain's batched
MemmapDataset loader, and computes per structure:

  atoms, periodic directions, cell volume, number density (atoms / A^3) and
  mass density (g / cm^3) for 3D periodic cells, energy (total and per atom),
  largest force, and the size of the full neighbour list at `--cutoff`;

and per atom: the force norm and the number of neighbours. Neighbour lists are
computed with vesin, in the DataLoader workers.

Writes to OUT (default `dataset-stats/`), all uncompressed `.npz` so they load
with a memory map (`np.load(path, mmap_mode="r")`):

  per-structure.npz  one array per quantity above, plus `split` (indexing
                     `splits`, the directory names) and `index` in the split
  per-atom.npz       per-atom histograms: `force_edges` / `force_counts`, and
                     `neighbour_counts[k]` = atoms with k neighbours
  histograms.npz     every distribution `plot_dataset_stats.py` draws, binned
                     once for each split and for `all` (same bins everywhere),
                     as `<split>/<name>/edges`, `.../counts` and `.../median`
  summary.txt        percentiles of everything, per split and over all

Plotting is separate and only reads histograms.npz:

  python plot_dataset_stats.py OUT/histograms.npz [--split train] [-o plot.png]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from ase.data import atomic_masses
from torch.utils.data import DataLoader
from vesin import NeighborList

from metatrain.utils.data import collate_batch
from metatrain.utils.data.dataset import MemmapDataset


AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660
# per-atom histograms: force norms on log bins (plus [0, 1e-4)), neighbour
# counts on integers
FORCE_EDGES = np.concatenate([[0.0], np.logspace(-4, 4, 161)])
MAX_NEIGHBOURS = 4096


def target_options(forces_key):
    forces = {"key": forces_key} if forces_key else False
    return {
        "energy": {
            "key": "energy",
            "quantity": "energy",
            "unit": "eV",
            "sample_kind": "system",
            "num_subtargets": 1,
            "type": "scalar",
            "forces": forces,
            "stress": False,
            "virial": False,
        }
    }


class Statistics:
    """Collate function: reduce a batch to its statistics, in the worker."""

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def __call__(self, samples):
        batch = collate_batch(samples, ["energy"])
        energy = batch.targets["energy"].block()
        forces = (
            -energy.gradient("positions").values[..., 0].numpy()
            if energy.has_gradient("positions")
            else None
        )

        n = len(batch.systems)
        stats = {k: np.full(n, np.nan) for k in ("volume", "number_density", "mass_density")}
        stats["n_atoms"] = np.empty(n, dtype=np.int64)
        stats["n_periodic"] = np.empty(n, dtype=np.int8)
        stats["n_pairs"] = np.empty(n, dtype=np.int64)
        stats["max_force"] = np.full(n, np.nan)
        stats["energy"] = energy.values[:, 0].numpy().copy()
        force_counts = np.zeros(len(FORCE_EDGES) - 1, dtype=np.int64)
        neighbour_counts = np.zeros(MAX_NEIGHBOURS + 1, dtype=np.int64)

        calculator = NeighborList(cutoff=self.cutoff, full_list=True)
        start = 0
        for k, system in enumerate(batch.systems):
            positions = system.positions.numpy()
            types = system.types.numpy()
            cell = system.cell.numpy()
            pbc = system.pbc.numpy()
            atoms = len(types)
            stats["n_atoms"][k] = atoms
            stats["n_periodic"][k] = pbc.sum()
            if pbc.all():
                volume = abs(np.linalg.det(cell))
                stats["volume"][k] = volume
                stats["number_density"][k] = atoms / volume
                stats["mass_density"][k] = (
                    atomic_masses[types].sum() / volume * AMU_PER_A3_TO_G_PER_CM3
                )

            first = calculator.compute(positions, cell, pbc, quantities="i")[0]
            stats["n_pairs"][k] = len(first)
            per_atom = np.bincount(first, minlength=atoms)
            neighbour_counts += np.bincount(
                np.minimum(per_atom, MAX_NEIGHBOURS), minlength=MAX_NEIGHBOURS + 1
            )

            if forces is not None:
                norms = np.linalg.norm(forces[start : start + atoms], axis=1)
                stats["max_force"][k] = norms.max(initial=0.0)
                force_counts += np.histogram(norms, FORCE_EDGES)[0]
            start += atoms

        stats["energy_per_atom"] = stats["energy"] / stats["n_atoms"]
        return stats, force_counts, neighbour_counts


def scan(directory, cutoff, batch_size, workers, forces_key, limit):
    forces_key = forces_key if (Path(directory) / f"{forces_key}.bin").exists() else None
    dataset = MemmapDataset(directory, target_options(forces_key))
    indices = range(len(dataset) if limit is None else min(limit, len(dataset)))
    loader = DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=batch_size,
        collate_fn=Statistics(cutoff),
        num_workers=workers,
        multiprocessing_context="fork" if workers > 0 else None,
    )
    parts, forces, neighbours = [], 0, 0
    for i, (stats, force_counts, neighbour_counts) in enumerate(loader):
        parts.append(stats)
        forces = forces + force_counts
        neighbours = neighbours + neighbour_counts
        if i % 50 == 0:
            print(f"\r{directory}: {sum(len(p['n_atoms']) for p in parts)}/{len(indices)}", end="", flush=True)
    print(f"\r{directory}: {len(indices)} structures")
    stats = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    return stats, forces, neighbours


# ---------------------------------------------------------------- histograms

def integer_log_edges(maximum, bins=70):
    """Log-spaced bins with edges between integers, so none is empty by construction."""
    centers = np.unique(np.round(np.logspace(0, np.log10(max(maximum, 1) + 1), bins)))
    return np.concatenate([[centers[0] - 0.5], centers + 0.5])


def log_edges(values, bins=80):
    values = values[np.isfinite(values) & (values > 0)]
    return np.logspace(np.log10(values.min()), np.log10(values.max()), bins + 1)


def central_edges(values, bins=80, tails=0.1):
    """Linear bins over the central (100 - 2 * tails) % of the values."""
    lo, hi = np.nanpercentile(values, [tails, 100 - tails])
    return np.linspace(lo, hi, bins + 1)


def histogram_bins(stats):
    """The binning of every per-structure panel, from the data of all splits."""
    finite = lambda key: stats[key][np.isfinite(stats[key])]  # noqa: E731
    ratio = stats["n_pairs"] / np.maximum(stats["n_atoms"], 1)
    return {
        "n_atoms": integer_log_edges(stats["n_atoms"].max()),
        "volume": log_edges(finite("volume")),
        "mass_density": central_edges(finite("mass_density")),
        "number_density": central_edges(finite("number_density")),
        "energy": central_edges(stats["energy"]),
        "energy_per_atom": central_edges(stats["energy_per_atom"]),
        "max_force": log_edges(finite("max_force")),
        # shifted by one so that empty lists show on a log axis
        "n_pairs": integer_log_edges(stats["n_pairs"].max() + 1),
        "size_vs_neighbours": (
            np.logspace(0, np.log10(stats["n_atoms"].max() + 1), 46),
            log_edges(ratio[ratio > 0], bins=45),
        ),
    }


def split_histograms(prefix, stats, forces, neighbours, bins):
    """Every panel for one split, as flat `<prefix>/<name>/<field>` arrays."""
    out = {}

    def put(name, edges, counts, values=None, **extra):
        out[f"{prefix}/{name}/edges"] = edges
        out[f"{prefix}/{name}/counts"] = counts
        if values is not None:
            values = values[np.isfinite(values)]
            out[f"{prefix}/{name}/median"] = np.median(values) if len(values) else np.nan
            # share of the values the bins leave out (tails of the linear panels)
            inside = (values >= edges[0]) & (values <= edges[-1])
            out[f"{prefix}/{name}/outside"] = 1 - inside.mean() if len(values) else 0.0
        for key, value in extra.items():
            out[f"{prefix}/{name}/{key}"] = value

    for name in ("n_atoms", "volume", "mass_density", "number_density", "energy", "energy_per_atom", "max_force"):
        values = stats[name].astype(float)
        put(name, bins[name], np.histogram(values[np.isfinite(values)], bins[name])[0], values)
    pairs = stats["n_pairs"].astype(float) + 1
    put("n_pairs", bins["n_pairs"], np.histogram(pairs, bins["n_pairs"])[0], pairs)
    put("n_periodic", np.arange(5) - 0.5, np.bincount(stats["n_periodic"], minlength=4))
    put("force_per_atom", FORCE_EDGES, forces)

    counts = np.arange(len(neighbours))
    cumulative = np.cumsum(neighbours) / max(neighbours.sum(), 1)
    put("neighbours_per_atom", np.arange(len(neighbours) + 1) - 0.5, neighbours,
        median=counts[np.searchsorted(cumulative, 0.5)])

    x_edges, y_edges = bins["size_vs_neighbours"]
    ok = (stats["n_atoms"] > 0) & (stats["n_pairs"] > 0)
    grid = np.histogram2d(stats["n_atoms"][ok], (stats["n_pairs"] / stats["n_atoms"].clip(1))[ok],
                          [x_edges, y_edges])[0]
    put("size_vs_neighbours", x_edges, grid, y_edges=y_edges)

    out[f"{prefix}/n_structures"] = len(stats["n_atoms"])
    out[f"{prefix}/n_atom_total"] = stats["n_atoms"].sum()
    return out


# ------------------------------------------------------------------- summary

PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)

DESCRIPTIONS = {
    "n_atoms": "atoms per structure",
    "n_pairs": "neighbour pairs per structure (full list)",
    "volume": "cell volume, 3D periodic [A^3]",
    "number_density": "number density, 3D periodic [atoms/A^3]",
    "mass_density": "mass density, 3D periodic [g/cm^3]",
    "energy": "energy [eV]",
    "energy_per_atom": "energy per atom [eV/atom]",
    "max_force": "largest force norm [eV/A]",
}


def histogram_percentiles(values, counts, q):
    cumulative = np.cumsum(counts) / counts.sum()
    return [values[min(np.searchsorted(cumulative, p / 100), len(counts) - 1)] for p in q]


def summarize(name, stats, forces, neighbours, cutoff):
    lines = [f"== {name}: {len(stats['n_atoms'])} structures, {stats['n_atoms'].sum()} atoms"]
    periodic = np.bincount(stats["n_periodic"], minlength=4)
    lines.append(
        "periodic directions: "
        + ", ".join(f"{d}D {periodic[d]} ({100 * periodic[d] / periodic.sum():.1f} %)" for d in range(4))
    )
    header = "".join(f"{f'p{p}':>12}" for p in PERCENTILES)
    lines.append(f"{'':44}{'mean':>12}{header}")
    for key, description in DESCRIPTIONS.items():
        values = stats[key][np.isfinite(stats[key])]
        if len(values) == 0:
            continue
        row = "".join(f"{v:12.5g}" for v in np.percentile(values, PERCENTILES))
        lines.append(f"{description:44}{values.mean():12.5g}{row}")
    if neighbours.sum():
        counts = np.arange(len(neighbours))
        row = "".join(f"{v:12.5g}" for v in histogram_percentiles(counts, neighbours, PERCENTILES))
        mean = (counts * neighbours).sum() / neighbours.sum()
        lines.append(f"{f'neighbours per atom within {cutoff} A':44}{mean:12.5g}{row}")
        lines.append(f"{'':44}isolated atoms (no neighbour): {neighbours[0]}")
    if np.sum(forces):
        centers = np.concatenate([[0.0], np.sqrt(FORCE_EDGES[1:-1] * FORCE_EDGES[2:])])
        row = "".join(f"{v:12.3g}" for v in histogram_percentiles(centers, forces, PERCENTILES))
        lines.append(f"{'force norm per atom [eV/A] (binned)':44}{'':12}{row}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directories", nargs="+")
    parser.add_argument("-o", "--output", default="dataset-stats")
    parser.add_argument("--cutoff", type=float, default=4.5, help="neighbour list cutoff [A], PET-MAD's by default")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--forces-key", default="non_conservative_force")
    parser.add_argument("--limit", type=int, default=None, help="only the first N structures of each directory")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    splits = [Path(d).name for d in args.directories]

    results = [
        scan(d, args.cutoff, args.batch_size, args.workers, args.forces_key, args.limit)
        for d in args.directories
    ]
    stats = {key: np.concatenate([r[0][key] for r in results]) for key in results[0][0]}
    forces = sum(r[1] for r in results)
    neighbours = sum(r[2] for r in results)

    np.savez(
        output / "per-structure.npz",
        **stats,
        split=np.concatenate([np.full(len(r[0]["n_atoms"]), i, dtype=np.int8) for i, r in enumerate(results)]),
        index=np.concatenate([np.arange(len(r[0]["n_atoms"])) for r in results]),
        splits=np.array(splits),
    )
    np.savez(output / "per-atom.npz", force_edges=FORCE_EDGES, force_counts=forces,
             neighbour_counts=neighbours, cutoff=args.cutoff)

    bins = histogram_bins(stats)
    histograms = split_histograms("all", stats, forces, neighbours, bins)
    for name, r in zip(splits, results):
        histograms.update(split_histograms(name, *r, bins))
    np.savez(output / "histograms.npz", **histograms, splits=np.array(["all", *splits]), cutoff=args.cutoff)

    sections = [summarize(name, *r, args.cutoff) for name, r in zip(splits, results)]
    if len(results) > 1:
        sections.append(summarize("all", stats, forces, neighbours, args.cutoff))
    summary = "\n\n".join(sections)
    (output / "summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"\nwrote {output}/{{per-structure.npz, per-atom.npz, histograms.npz, summary.txt}}")
    print(f"plot with: python {Path(__file__).with_name('plot_dataset_stats.py')} {output / 'histograms.npz'}")


if __name__ == "__main__":
    main()
