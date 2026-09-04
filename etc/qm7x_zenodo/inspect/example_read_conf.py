#!/usr/bin/env python3
"""Load one QM7-X conformation the same way the converter does.

Prints geometry, the mapped endpoints, and (when --tmap) the TensorMap
layouts that metatrain trains on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from zenodo_to_metatensor import (  # noqa: E402
    iter_confs,
    load_structure,
    to_targets,
)


def _first_conf(h5: h5py.File, mol: str | None, conf: str | None):
    if mol and conf:
        return mol, conf, h5[mol][conf]
    for mol_id, conf_id, group in iter_confs(h5):
        if mol and mol_id != mol:
            continue
        return mol_id, conf_id, group
    raise SystemExit("no conformation groups found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--mol", default=None, help="molecule id, e.g. 1")
    parser.add_argument("--conf", default=None, help="conformation id")
    parser.add_argument(
        "--tmap",
        action="store_true",
        help="also print TensorMap keys / block shapes",
    )
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as h5:
        mol_id, conf_id, group = _first_conf(h5, args.mol, args.conf)
        entry = load_structure(group)
    if entry is None:
        raise SystemExit(f"{mol_id}/{conf_id} is missing atNUM/atXYZ/eAT")

    print(f"{mol_id}/{conf_id}")
    print(f"  Z        {entry['numbers'].tolist()}")
    print(f"  N atoms  {len(entry['numbers'])}")
    print(f"  eAT      {entry['energy']:.6f} eV")
    if entry["forces"] is not None:
        rms = float(np.sqrt((entry["forces"] ** 2).mean()))
        print(f"  totFOR   rms {rms:.6f} eV/Å  shape {entry['forces'].shape}")
    if entry["hlgap"] is not None:
        print(f"  HLgap    {entry['hlgap']:.6f} eV")
    if entry["dipole"] is not None:
        print(f"  vDIP     {entry['dipole']}")
    if entry["hirshfeld_charge"] is not None:
        print(f"  hCHG     {entry['hirshfeld_charge']}")
    if entry["polarizability"] is not None:
        print(f"  mTPOL\n{entry['polarizability']}")

    if args.tmap:
        print()
        for name, tmap in to_targets(entry, system=0).items():
            print(f"TensorMap {name!r}  keys={tmap.keys}")
            for key, block in tmap.items():
                print(
                    f"  {key}: samples={block.samples.names} "
                    f"values={tuple(block.values.shape)} "
                    f"grads={list(block.gradients_list())}"
                )


if __name__ == "__main__":
    main()
