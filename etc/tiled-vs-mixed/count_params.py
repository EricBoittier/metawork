import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
import bench
from model_bench import PETLRBench

CACHE = {"mad": "/tvm_pool.pkl", "oc25": "/tvm_oc25.pkl"}

def count(tree):
    return sum(int(np.prod(x.shape)) for x in jax.tree.leaves(tree))

for cfg in ("mad", "oc25"):
    bench.select_config(cfg)
    path = os.environ["SCRATCH"] + CACHE[cfg]
    if not os.path.exists(path):
        print(f"[{cfg}] no pool cache, skipped"); continue
    with open(path, "rb") as f:
        blob = pickle.load(f)
    samples, props = blob["samples"], blob["properties"]

    print(f"\n===== {cfg} ({bench.CONFIGS[cfg]['folder']}) =====")
    for backend in ("tiled", "mixed"):
        m = PETLRBench(backend=backend, lr=True, **bench.MODEL)
        p = bench.init_params(m, props, samples)
        tot = count(p)
        sr = count(p["params"]["sr"])
        lr = count(p["params"]["lr"])
        print(f"  {backend:>5}: total {tot:>12,}   sr {sr:>12,}   lr {lr:>12,}")
        if backend == "tiled":
            ref = jax.tree.map(lambda x: x.shape, p)
    m0 = PETLRBench(backend="tiled", lr=False, **bench.MODEL)
    p0 = bench.init_params(m0, props, samples, sr_only=True)
    print(f"  sr-only: total {count(p0):>12,}")
    print("  LR sub-module breakdown (tiled):")
    m = PETLRBench(backend="tiled", lr=True, **bench.MODEL)
    p = bench.init_params(m, props, samples)
    for k, v in sorted(p["params"]["lr"].items()):
        print(f"    {k:<28} {count(v):>10,}")
