import gzip
import re
from collections import defaultdict

path = "/Volumes/ERIC/data/mad.data/madcore_0-2pow18.extxyz.gz"
out_path = "mad_subset.xyz"
KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
PER_CLASS = 1000
ZBL_MAX = 50.0

counts = defaultdict(int)
n_written = 0

with gzip.open(path, "rt") as handle, open(out_path, "w") as out:
    while True:
        n_line = handle.readline()
        if not n_line:
            break
        n_atoms = int(n_line)
        comment = handle.readline()
        atom_lines = [handle.readline() for _ in range(n_atoms)]

        header = dict(KV.findall(comment))
        if float(header["zbl_max_force"]) > ZBL_MAX:
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
        key = (kind, size)
        if counts[key] >= PER_CLASS:
            continue
        counts[key] += 1
        out.write(n_line)
        out.write(comment)
        out.writelines(atom_lines)
        n_written += 1

for key, n in sorted(counts.items()):
    print(f"{key}: wrote {n}")
print(n_written, "structures")
