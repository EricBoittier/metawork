import os
import numpy as np
from pathlib import Path
from collections import Counter
from mmap_ninja import RaggedMmap

folder = Path(os.environ["DATASETS"]) / "MAD/v1.6/train"
print("folder:", folder)
print("side files:", sorted(p.name for p in folder.iterdir()))
print("has info.yaml:", (folder / "info.yaml").is_file())

mm = RaggedMmap(folder / "mmap")
n = len(mm)
print("records:", n)

starts = np.asarray(mm.starts) if hasattr(mm, "starts") else None
print("has starts:", starts is not None)

na = np.zeros(n, dtype=np.int64)
pbc = np.zeros(n, dtype=np.int64)
for i in range(n):
    head = mm[i][:4]
    na[i] = int(head[0])
    pbc[i] = int(head[1:4].sum())

periodic = pbc > 0
print("\n== FULL TRAIN SET (n=%d) ==" % n)
print("pbc dims:", dict(sorted(Counter(pbc.tolist()).items())))
print("periodic fraction: %.4f" % periodic.mean())

def describe(tag, arr):
    if not len(arr):
        return
    print("\n-- %s (n=%d) --" % (tag, len(arr)))
    print("  mean %.1f" % arr.mean())
    for q in (50, 90, 99, 99.9, 100):
        print("  p%-5s %6.0f" % (q, np.percentile(arr, q)))
    print("  max/mean ratio: %.1f" % (arr.max() / arr.mean()))

describe("all structures", na)
describe("periodic structures", na[periodic])

print("\n-- how max scales with pool size (periodic only, seed 0) --")
rng = np.random.default_rng(0)
pna = na[periodic]
for size in (256, 1024, 2048, 8192, 32768, len(pna)):
    if size > len(pna):
        continue
    draws = [pna[rng.choice(len(pna), size=size, replace=False)].max() for _ in range(5)]
    print("  pool %-7d periodic-max %5.0f  -> mixed padding %.1fx" % (
        size, np.mean(draws), np.mean(draws) / pna.mean()))
