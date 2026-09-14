import os, sys
import numpy as np
sys.path.insert(0, os.environ["HOME"] + "/shims/petjax-compat")
from marathon.grain import DataSource

src = DataSource(os.environ["DATASETS"] + "/MAD/v1.6/train")
print("n_structures:", len(src))
rng = np.random.default_rng(0)
idx = rng.choice(len(src), size=400, replace=False)
na, pbc = [], []
for i in idx:
    a = src.get_atoms(int(i))
    na.append(len(a))
    pbc.append(int(a.get_pbc().sum()))
na = np.array(na); pbc = np.array(pbc)
print("atoms: min %d  median %d  mean %.1f  p90 %d  max %d" % (na.min(), np.median(na), na.mean(), np.percentile(na,90), na.max()))
print("pbc dims histogram:", {int(k): int(v) for k, v in zip(*np.unique(pbc, return_counts=True))})
import jax
print("jax", jax.__version__, jax.devices())
