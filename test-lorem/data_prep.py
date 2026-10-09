import gzip
import re
from collections import defaultdict

import numpy as np
from ase.io import iread, write

path = "/Volumes/ERIC/data/mad.data/madcore_0-2pow18.extxyz.gz"
KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
PER_CLASS = 30
ZBL_MAX = 50.0  # drop frames with very close contacts

buckets = defaultdict(list)
with gzip.open(path, "rt") as handle:
    index = 0
    while True:
        line = handle.readline()
        if not line:
            break
        n_atoms = int(line)
        header = dict(KV.findall(handle.readline()))
        for _ in range(n_atoms):
            handle.readline()
        if float(header["zbl_max_force"]) > ZBL_MAX:
            index += 1
            continue
        kind = {"F F F": "molecule", "T T T": "crystal", "T T F": "slab"}[
            header["pbc"].strip('"')
        ]
        size = (
            "1-10" if n_atoms <= 10
            else "11-30" if n_atoms <= 30
            else "31-100" if n_atoms <= 100
            else ">100"
        )
        # optional third axis, e.g. only these groups:
        # if header["dataset_group"] not in {"OMol25", "ANI-1", "Alexandria", "OMat24"}:
        #     index += 1
        #     continue
        buckets[(kind, size)].append(index)
        index += 1

chosen = []
for key, indices in sorted(buckets.items()):
    take = indices[:PER_CLASS]
    chosen.extend(take)
    print(f"{key}: {len(indices)} available, taking {len(take)}")

chosen = set(chosen)
frames = []
for index, atoms in enumerate(iread(path, format="extxyz")):
    if index in chosen:
        frames.append(atoms)
    if len(frames) == len(chosen):
        break
write("mad_subset.xyz", frames)
print(len(frames), "structures")
