"""Markdown tables (ms per call, and per atom-step) from the CSV of run.sh."""
import csv
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
sizes = sorted({int(r["n_atoms"]) for r in rows})
table = defaultdict(dict)
for r in rows:
    table[(r["mode"], r["wrapper"], r["backend"], r["threads"])][int(r["n_atoms"])] = float(r["ms"])

for mode in ("energy", "forces", "nc"):
    print(f"\n### {mode} (ms per call)\n")
    print("| wrapper | backend | threads | " + " | ".join(f"{n} atoms" for n in sizes) + " |")
    print("| --- | --- | ---: | " + " | ".join("---:" for _ in sizes) + " |")
    for (m, wrapper, backend, threads), times in sorted(table.items()):
        if m == mode:
            cells = " | ".join(f"{times[n]:.1f}" if n in times else "" for n in sizes)
            print(f"| {wrapper} | {backend} | {threads} | {cells} |")
