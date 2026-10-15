"""Per-row dataset_group and dataset_id of every MAD-CORE structure.

usage: python extract_groups.py [DATA_DIR] [-o groups.npz]

Streams the headers of every `madcore_*.extxyz.gz` shard (one `zcat | grep`
per shard, in parallel) and writes, in global row order (= fps_order):

  group     int16 code of dataset_group, indexing `groups`
  dataset   int16 code of dataset_id, indexing `datasets`
  groups, datasets   the names
  zbl_energy_per_atom, zbl_max_force, zbl_max_atomic_energy   float32, the
            ZBL repulsion header keys (eV/atom, eV/A, eV)

It checks that row i of each shard has fps_order i, i.e. that feature row i
(madcore_features_*.h5) and frame i describe the same structure.
"""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


ZBL = ("zbl_energy_per_atom", "zbl_max_force", "zbl_max_atomic_energy")


def shard_labels(path):
    command = f"zcat '{path}' | grep -aoE '(dataset_group|dataset_id|fps_order|{'|'.join(ZBL)})=[^ ]+'"
    out = subprocess.run(command, shell=True, check=True, capture_output=True, text=True).stdout.split()
    keys = [line.split("=", 1) for line in out]
    groups = [v for k, v in keys if k == "dataset_group"]
    datasets = [v for k, v in keys if k == "dataset_id"]
    order = np.array([int(v) for k, v in keys if k == "fps_order"], dtype=np.int64)
    zbl = {z: np.array([float(v) for k, v in keys if k == z], dtype=np.float32) for z in ZBL}
    assert len(groups) == len(datasets) == len(order) and all(len(v) == len(order) for v in zbl.values()), path
    return groups, datasets, order, zbl


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", nargs="?", default=Path("~/data/madcore").expanduser(), type=Path)
    parser.add_argument("-o", "--output", default=Path(__file__).with_name("groups.npz"), type=Path)
    args = parser.parse_args()

    files = json.loads((args.data / "index.json").read_text())["files"]
    shards = sorted(
        ((m["rows"], args.data / name) for name, m in files.items() if m["family"] == "xyz"),
        key=lambda item: item[0][0],
    )
    with ThreadPoolExecutor(len(shards)) as pool:
        results = list(pool.map(lambda s: shard_labels(s[1]), shards))

    groups, datasets, zbl = [], [], {z: [] for z in ZBL}
    for ((start, stop), path), (g, d, order, zb) in zip(shards, results):
        assert np.array_equal(order, np.arange(start, stop)), f"{path}: rows are not in fps_order"
        groups += g
        datasets += d
        for z in ZBL:
            zbl[z].append(zb[z])
        print(f"{path.name}: {len(g)} rows")

    group_names, group_codes = np.unique(np.array(groups), return_inverse=True)
    dataset_names, dataset_codes = np.unique(np.array(datasets), return_inverse=True)
    np.savez(args.output, group=group_codes.astype(np.int16), dataset=dataset_codes.astype(np.int16),
             groups=group_names, datasets=dataset_names, **{z: np.concatenate(v) for z, v in zbl.items()})
    counts = np.bincount(group_codes)
    for k in np.argsort(counts)[::-1]:
        print(f"{group_names[k]:40s} {counts[k]:>10,}")
    print(f"wrote {args.output}: {len(group_codes):,} rows")


if __name__ == "__main__":
    main()
