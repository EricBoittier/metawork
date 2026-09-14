import os
import numpy as np
from collections import Counter
from marathon.grain import DataSource

src = DataSource(os.environ["DATASETS"] + "/oc25/train")
print("total structures:", len(src))
a0 = src.get_atoms(0)
print("info keys:", sorted(a0.info.keys()))
print("pbc of first:", a0.get_pbc(), "natoms", len(a0))
print("cell:\n", np.asarray(a0.get_cell()))

rng = np.random.default_rng(0)
idx = rng.choice(len(src), size=4000, replace=False)
na, pbc = [], []
for i in idx:
    a = src.get_atoms(int(i))
    na.append(len(a)); pbc.append(int(a.get_pbc().sum()))
na = np.array(na); pbc = np.array(pbc)

print("\n-- sample n=%d --" % len(na))
print("  mean %.1f" % na.mean())
for q in (0, 25, 50, 75, 90, 99, 100):
    print("  p%-3d %6.0f" % (q, np.percentile(na, q)))
print("  pbc dims:", dict(sorted(Counter(pbc.tolist()).items())))
per = na[pbc > 0]
if len(per):
    print("  periodic: n=%d mean %.1f max %d  -> max/mean %.1f" % (
        len(per), per.mean(), per.max(), per.max() / per.mean()))
