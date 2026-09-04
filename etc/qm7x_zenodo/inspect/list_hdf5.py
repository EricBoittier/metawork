#!/usr/bin/env python3
"""Print molecule/conformation counts and dataset keys for a QM7-X HDF5 shard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from zenodo_to_metatensor import REQUIRED_HDF5, iter_confs  # noqa: E402

MAPPED = {
    "atNUM": "geometry",
    "atXYZ": "geometry",
    "eAT": "energy",
    "totFOR": "energy positions gradient",
    "HLgap": "mtt::hlgap",
    "vDIP": "mtt::dipole",
    "hCHG": "mtt::hirshfeld_charge",
    "hVDIP": "mtt::hirshfeld_dipole",
    "mTPOL": "mtt::polarizability",
    "mC6": "mtt::c6 (optional)",
    "atC6": "mtt::atomic_c6 (optional)",
}


def _shape(dataset) -> str:
    array = np.asarray(dataset[()])
    return f"{array.shape} {array.dtype}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument(
        "--max-confs",
        type=int,
        default=1,
        help="how many conformations to print keys for (default: 1)",
    )
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as h5:
        n_mol = sum(1 for _, obj in h5.items() if isinstance(obj, h5py.Group))
        n_conf = 0
        n_full = 0
        shown = 0
        print(f"{args.hdf5}: counting…", flush=True)
        for mol_id, conf_id, group in iter_confs(h5):
            n_conf += 1
            if all(k in group for k in REQUIRED_HDF5):
                n_full += 1
            if shown < args.max_confs:
                if shown == 0:
                    print()
                print(f"{mol_id}/{conf_id}")
                for key in group.keys():
                    mapped = MAPPED.get(key, "")
                    flag = f"  -> {mapped}" if mapped else ""
                    print(f"  {key:12s} {_shape(group[key])}{flag}")
                missing = [k for k in REQUIRED_HDF5 if k not in group]
                if missing:
                    print(f"  missing required keys: {missing}")
                shown += 1
        print()
        print(f"{args.hdf5}: {n_mol} molecules, {n_conf} conformations")
        print(f"  with the converter's full endpoint set: {n_full}")


if __name__ == "__main__":
    main()
