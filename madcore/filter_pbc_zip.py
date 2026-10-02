"""Split MAD-CORE DiskDataset zips by periodicity, dropping slabs.

LOREM's long-range block cannot handle partially periodic systems (slabs), so
every subset keeps only fully periodic (3D) or fully non-periodic (0D)
structures:

    mixed        3D + 0D
    periodic     3D
    nonperiodic  0D

Periodicity comes from stats-2pow18/per-structure.npz (``n_periodic``), whose
``index`` is the entry number inside each split's zip. Members are copied
byte for byte and renumbered; ``metadata/atom_counts.npy`` is rewritten.
``--limit N`` keeps only the first N kept train structures (timing subsets).

usage: python filter_pbc_zip.py SRC OUT [--limit N] [--subsets mixed periodic nonperiodic]
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
KEEP = {"mixed": (0, 3), "periodic": (3,), "nonperiodic": (0,)}


def periodicity(stats_path):
    stats = np.load(stats_path)
    splits = [str(s) for s in stats["splits"]]
    out = {}
    for k, name in enumerate(splits):
        mask = stats["split"] == k
        order = np.argsort(stats["index"][mask])
        index = stats["index"][mask][order]
        assert (index == np.arange(len(index))).all(), name
        out[name] = (stats["n_periodic"][mask][order], stats["n_atoms"][mask][order])
    return out


def write_subset(src, dst, keep, n_atoms):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst.with_suffix(".tmp"), "w", zipfile.ZIP_STORED) as zout:
        for new, old in enumerate(keep):
            for member in ("system.mta", "energy.mts"):
                zout.writestr(f"{new}/{member}", zin.read(f"{old}/{member}"))
        counts = io.BytesIO()
        np.save(counts, n_atoms[keep].astype(np.int64))
        zout.writestr("metadata/atom_counts.npy", counts.getvalue())
    dst.with_suffix(".tmp").rename(dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="directory with train/val/test.zip")
    parser.add_argument("out", type=Path)
    parser.add_argument("--stats", type=Path, default=HERE / "stats-2pow18" / "per-structure.npz")
    parser.add_argument("--subsets", nargs="+", default=list(KEEP), choices=list(KEEP))
    parser.add_argument("--limit", type=int, help="first N kept train structures only")
    args = parser.parse_args()

    info = periodicity(args.stats)
    for subset in args.subsets:
        name = subset if args.limit is None else f"{subset}-{args.limit}"
        for split, (n_periodic, n_atoms) in info.items():
            keep = np.flatnonzero(np.isin(n_periodic, KEEP[subset]))
            if split == "train" and args.limit is not None:
                keep = keep[: args.limit]
            dst = args.out / name / f"{split}.zip"
            if dst.exists():
                print(f"{dst} exists, skipping")
                continue
            write_subset(args.src / f"{split}.zip", dst, keep, n_atoms)
            print(f"{name}/{split}: {len(keep)} of {len(n_periodic)} structures, {n_atoms[keep].sum()} atoms")


if __name__ == "__main__":
    main()
