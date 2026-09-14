import os
import numpy as np
from collections import Counter
from marathon.grain import DataSource

src = DataSource(os.environ["DATASETS"] + "/MAD/v1.6/train")
print("total structures:", len(src))

a0 = src.get_atoms(0)
print("info keys:", sorted(a0.info.keys()))
print("arrays keys:", sorted(a0.arrays.keys()))
print("example info:", {k: a0.info[k] for k in list(a0.info)[:8]})

rng = np.random.default_rng(1)
idx = rng.choice(len(src), size=6000, replace=False)

key = None
for cand in ("subset", "config_type", "name", "dataset", "origin", "structure_type"):
    if cand in a0.info:
        key = cand
        break
print("subset key:", key)

na, pbc, sub = [], [], []
for i in idx:
    a = src.get_atoms(int(i))
    na.append(len(a))
    pbc.append(int(a.get_pbc().sum()))
    sub.append(str(a.info.get(key, "?")) if key else "?")

na = np.array(na); pbc = np.array(pbc)
print("\n-- overall (n=%d) --" % len(na))
for q in (0, 5, 25, 50, 75, 90, 95, 99, 100):
    print("  p%-3d %6.0f" % (q, np.percentile(na, q)))
print("  mean %.1f" % na.mean())
print("  fraction with <=4 atoms: %.3f" % (na <= 4).mean())
print("  fraction with >=50 atoms: %.3f" % (na >= 50).mean())
print("  pbc dims:", dict(Counter(pbc.tolist())))

print("\n-- periodic only (these drive the Ewald padding) --")
pna = na[pbc > 0]
if len(pna):
    for q in (50, 90, 99, 100):
        print("  p%-3d %6.0f" % (q, np.percentile(pna, q)))
    print("  mean %.1f  n=%d" % (pna.mean(), len(pna)))

if key:
    print("\n-- by subset --")
    c = Counter(sub)
    print("  %-28s %7s %7s %7s %7s" % ("subset", "frac", "mean_na", "max_na", "pbc%"))
    for name, n in c.most_common(25):
        m = np.array([s == name for s in sub])
        print("  %-28s %7.3f %7.1f %7d %7.2f" % (name, n / len(sub), na[m].mean(), na[m].max(), (pbc[m] > 0).mean()))
