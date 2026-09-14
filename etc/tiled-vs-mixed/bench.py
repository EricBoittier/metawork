"""tiled vs mixed Ewald batching: max batch size and time per sample.

Two phases, run as separate processes so an OOM in one trial cannot poison
the next:

  prep   build a fixed pool of samples once, pickle it
  trial  one (backend, batch_size) point: build the batch, jit a full
         forward+backward+Adam step, time it

Dataset/model presets live in CONFIGS; pick one with --config.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import argparse
import json
import pickle
import time

DATASETS = os.environ.get(
    "DATASETS", "/capstor/store/cscs/swissai/aa002/rumiants/datasets"
)

CONFIGS = {
    # matches work/MAD/lr_MAD_v16_xs_nn-16_scratch_gnn1
    "mad": dict(
        folder="MAD/v1.6/train",
        keys=("energy", "forces", "stress"),
        num_k=4096,
        BM=8,
        BK=128,
        loss_weights={"energy": 10.0, "forces": 1.0, "stress": 1.0},
        num_neighbors_adaptive=16,
        model=dict(
            adaptive_cutoff_method="solver",
            attention_temperature=1.0,
            cutoff=7.5,
            cutoff_width=2.0,
            cutoff_width_adaptive=1.0,
            d_feedforward=352,
            d_head=128,
            d_node=512,
            d_pet=192,
            max_atomic_number=102,
            num_attention_layers=1,
            num_gnn_layers=1,
            num_heads=12,
            num_charges=8,
            lr_scale_init=1.0,
        ),
    ),
    # Local, reduced-scale stand-in for "mad"/"oc25": neither of those datasets
    # (nor the CSCS /capstor paths, nor SLURM/multi-GPU) are available on this
    # machine, so this points at a small marathon.grain DataSource built from
    # this project's own periodic bio_dimers dataset (see
    # ../../analysis/etc.../prep_local_bio_dataset.py-equivalent prep step),
    # with a smaller num_k (bio_dimers' 30 A cell doesn't need MAD's 4096) so
    # a full sweep is tractable on a single 16GB GPU. See tiled_vs_mixed's
    # README/REPORT for how this differs from the original SLURM sweep.
    "local_bio": dict(
        folder="local/bio_dimers",
        keys=("energy", "forces"),
        num_k=1024,
        BM=8,
        BK=128,
        loss_weights={"energy": 1.0, "forces": 1.0},
        num_neighbors_adaptive=16,
        model=dict(
            adaptive_cutoff_method="solver",
            attention_temperature=1.0,
            cutoff=7.5,
            cutoff_width=2.0,
            cutoff_width_adaptive=1.0,
            d_feedforward=256,
            d_head=128,
            d_node=512,
            d_pet=128,
            max_atomic_number=102,
            num_attention_layers=1,
            num_gnn_layers=2,
            num_heads=8,
            num_charges=8,
            lr_scale_init=1.0,
        ),
    ),
    # matches work/OC25/oc25_xs_nn-16_lr_cons
    "oc25": dict(
        folder="oc25/train",
        keys=("energy", "forces"),
        num_k=4096,
        BM=8,
        BK=128,
        loss_weights={"energy": 1.0, "forces": 1.0},
        num_neighbors_adaptive=16,
        model=dict(
            adaptive_cutoff_method="solver",
            attention_temperature=1.0,
            cutoff=7.5,
            cutoff_width=2.0,
            cutoff_width_adaptive=1.0,
            d_feedforward=256,
            d_head=128,
            d_node=512,
            d_pet=128,
            max_atomic_number=102,
            num_attention_layers=1,
            num_gnn_layers=2,
            num_heads=8,
            num_charges=8,
            lr_scale_init=1.0,
        ),
    ),
}

CONFIG_NAME = "mad"
FOLDER = KEYS = NUM_K = BM = BK = LOSS_WEIGHTS = MODEL = TO_SAMPLE = None


def select_config(name):
    """Bind the module-level config globals to one entry of CONFIGS."""
    global CONFIG_NAME, FOLDER, KEYS, NUM_K, BM, BK, LOSS_WEIGHTS, MODEL, TO_SAMPLE
    c = CONFIGS[name]
    CONFIG_NAME = name
    FOLDER = c["folder"]
    KEYS = c["keys"]
    NUM_K = c["num_k"]
    BM, BK = c["BM"], c["BK"]
    LOSS_WEIGHTS = c["loss_weights"]
    MODEL = c["model"]
    TO_SAMPLE = dict(
        cutoff=MODEL["cutoff"],
        cutoff_width_adaptive=MODEL["cutoff_width_adaptive"],
        adaptive_cutoff_method=MODEL["adaptive_cutoff_method"],
        num_neighbors_adaptive=c["num_neighbors_adaptive"],
    )
    return c


select_config("mad")


# -- prep ----------------------------------------------------------------------

_worker = {}


def _init_worker(folder, properties):
    from marathon.grain import DataSource

    from iris.pet.sample import AdaptiveToSample

    _worker["source"] = DataSource(folder)
    _worker["to_sample"] = AdaptiveToSample(
        keys=KEYS,
        properties=properties,
        with_lr=True,
        num_k=NUM_K,
        **TO_SAMPLE,
    )


def _prep_one(index):
    return _worker["to_sample"].map(_worker["source"].get_atoms(int(index)))


def prep(args):
    import multiprocessing as mp

    from marathon.grain import DataSource

    folder = f"{DATASETS}/{FOLDER}"
    source = DataSource(folder)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(source), size=args.pool, replace=False)

    t0 = time.perf_counter()
    with mp.Pool(
        args.workers, initializer=_init_worker, initargs=(folder, source.properties)
    ) as pool:
        samples = pool.map(_prep_one, indices, chunksize=4)
    print(f"prepped {len(samples)} samples in {time.perf_counter() - t0:.1f}s", flush=True)

    with open(args.cache, "wb") as f:
        pickle.dump({"samples": samples, "properties": source.properties}, f, protocol=4)
    print(f"wrote {args.cache} ({os.path.getsize(args.cache) / 1e9:.2f} GB)", flush=True)

    n_atoms = np.array([len(s.lr_structure["positions"]) for s in samples])
    n_pbc = sum(hasattr(s.lr_structure["lr"], "k_grid") for s in samples)
    print(
        f"atoms/structure: mean {n_atoms.mean():.1f} median {np.median(n_atoms):.0f} "
        f"max {n_atoms.max()}; periodic {n_pbc}/{len(samples)}",
        flush=True,
    )


# -- batch construction --------------------------------------------------------


def components(backend, properties, sr_only=False):
    from iris.pet.batching import SR, TiledEwaldLR, token_space

    from mixed_lr import MixedEwaldLR

    sr = SR(keys=KEYS, properties=properties)
    lr = TiledEwaldLR(BM=BM, BK=BK) if backend == "tiled" else MixedEwaldLR()
    disc = {
        "n_structures": None,
        "n_atoms": None,
        "n_pairs": None,
        "sr.n_pairs": None,
        "sr.k_sel": token_space("multiples_of_2"),
        "lr.n_pairs": None,
        "lr.n_structures_pbc": None,
        "lr.n_pairs_nonpbc": None,
        "lr.n_atoms_pbc": f"multiples_of_{BM}" if backend == "tiled" else None,
        "lr.num_k": f"multiples_of_{BK}" if backend == "tiled" else None,
    }
    if sr_only:
        return {"sr": sr}, disc
    return {"sr": sr, "lr": lr}, disc


def build_batch(samples, batch_size, backend, properties, sr_only=False):
    """Budget from every chunk of `batch_size` in the pool; materialize chunk 0."""
    from iris.shared.megabatch import (
        component_arrays as component_view_arrays,
        component_view,
        merge_arrays,
        merge_shapes,
        merge_specs,
        min_shapes,
        padded_shapes,
    )

    comps, disc = components(backend, properties, sr_only=sr_only)
    spec = merge_specs({n: c.spec for n, c in comps.items()})

    chunks = [
        samples[i : i + batch_size]
        for i in range(0, len(samples) - batch_size + 1, batch_size)
    ]
    per_chunk = []
    for chunk in chunks:
        shapes = [merge_shapes({n: c.sizer(s) for n, c in comps.items()}) for s in chunk]
        per_chunk.append(min_shapes(shapes, spec))
    budget = {k: max(c[k] for c in per_chunk) for k in spec}
    budget = padded_shapes(budget, {k: disc[k] for k in budget})
    budget = {k: max(v, 1) for k, v in budget.items()}

    parts = {}
    for name, comp in comps.items():
        view = component_view(budget, name, comp.spec.keys())
        if hasattr(comp, "validate"):
            comp.validate(view)
        parts[name] = comp.materialize(chunks[0], view)

    shared = set()
    for comp in comps.values():
        shared.update(comp.shared_arrays)
    batch = merge_arrays(parts, shared)

    lr_real = lr_total = {}
    if "lr" in comps:
        lr_view = component_view_arrays(batch, "lr", comps["lr"].shared_arrays)
        lr_real, lr_total, _ = comps["lr"].info(lr_view)
    stats = {
        "budget": {k: int(v) for k, v in budget.items()},
        "n_chunks": len(chunks),
        "real_atoms": int(batch["atom_mask"].sum()),
        "real_pairs": int(batch["sr"]["pair_mask"].sum()),
        "lr_real": {k: int(v) for k, v in lr_real.items()},
        "lr_total": {k: int(v) for k, v in lr_total.items()},
    }
    return batch, stats


# -- trial ---------------------------------------------------------------------


def init_params(model, properties, samples, sr_only=False):
    import jax

    from model_bench import model_inputs

    tiny, _ = build_batch(samples[:4], 2, model.backend, properties, sr_only=sr_only)
    tiny = jax.tree.map(jax.numpy.asarray, tiny)
    truncated, lr, _ = model_inputs(model, tiny)
    return model.init(jax.random.key(0), truncated, lr)


def make_step(model, tx):
    import jax
    import jax.numpy as jnp
    import optax

    from model_bench import predict

    def masked_mse(pred, target, mask):
        err = (pred - target) ** 2
        m = mask.astype(err.dtype)
        while m.ndim < err.ndim:
            m = m[..., None]
        return (err * m).sum() / jnp.maximum(m.sum(), 1.0)

    want_stress = "stress" in KEYS

    def loss_fn(params, batch):
        out = predict(model, params, batch, stress=want_stress)
        labels = batch["labels"]
        loss = LOSS_WEIGHTS["energy"] * masked_mse(
            out["energy"],
            jnp.reshape(labels["energy"], (-1,)),
            jnp.reshape(labels["energy_mask"], (-1,)),
        )
        loss += LOSS_WEIGHTS["forces"] * masked_mse(
            out["forces"], labels["forces"], labels["forces_mask"]
        )
        if want_stress:
            loss += LOSS_WEIGHTS["stress"] * masked_mse(
                out["stress"], labels["stress"], labels["stress_mask"]
            )
        return loss

    @jax.jit
    def step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return step


def trial(args):
    import jax
    import optax

    from model_bench import PETLRBench

    jax.config.update("jax_default_matmul_precision", "float32")
    if args.compilation_cache:
        jax.config.update("jax_compilation_cache_dir", args.compilation_cache)

    with open(args.cache, "rb") as f:
        blob = pickle.load(f)
    samples, properties = blob["samples"], blob["properties"]

    model = PETLRBench(backend=args.backend, lr=not args.sr_only, **MODEL)
    params = init_params(model, properties, samples, sr_only=args.sr_only)

    batch, stats = build_batch(
        samples, args.batch_size, args.backend, properties, sr_only=args.sr_only
    )
    batch = jax.tree.map(jax.numpy.asarray, batch)

    tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(2e-4))
    opt_state = tx.init(params)
    step = make_step(model, tx)

    mem = {}
    try:
        compiled = step.lower(params, opt_state, batch).compile()
        ma = compiled.memory_analysis()
        mem = {
            "temp_gb": ma.temp_size_in_bytes / 1e9,
            "argument_gb": ma.argument_size_in_bytes / 1e9,
            "output_gb": ma.output_size_in_bytes / 1e9,
            "generated_code_gb": getattr(ma, "generated_code_size_in_bytes", 0) / 1e9,
        }
        mem["total_gb"] = mem["temp_gb"] + mem["argument_gb"] + mem["output_gb"]
    except Exception as exc:
        mem = {"error": repr(exc)[:200]}
    print(
        "MEMINFO "
        + json.dumps(
            {
                "config": CONFIG_NAME,
                "backend": args.backend,
                "sr_only": args.sr_only,
                "batch_size": args.batch_size,
                **mem,
            }
        ),
        flush=True,
    )

    t0 = time.perf_counter()
    params, opt_state, loss = step(params, opt_state, batch)
    jax.block_until_ready(loss)
    compile_time = time.perf_counter() - t0

    for _ in range(args.warmup):
        params, opt_state, loss = step(params, opt_state, batch)
    jax.block_until_ready(loss)

    times = []
    for _ in range(args.steps):
        t = time.perf_counter()
        params, opt_state, loss = step(params, opt_state, batch)
        jax.block_until_ready(loss)
        times.append(time.perf_counter() - t)

    times = np.array(times)
    stats_dev = jax.local_devices()[0].memory_stats() or {}
    result = {
        "config": CONFIG_NAME,
        "sr_only": args.sr_only,
        "BM": BM,
        "BK": BK,
        "backend": args.backend,
        "batch_size": args.batch_size,
        "ok": True,
        "step_s_mean": float(times.mean()),
        "step_s_std": float(times.std()),
        "step_s_min": float(times.min()),
        "s_per_sample": float(times.mean() / args.batch_size),
        "s_per_atom": float(times.mean() / stats["real_atoms"]),
        "compile_s": compile_time,
        "peak_gb": float(stats_dev.get("peak_bytes_in_use", 0)) / 1e9,
        "mem_analysis": mem,
        "loss": float(loss),
        **stats,
    }
    print("RESULT " + json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prep", "trial"])
    p.add_argument("--config", choices=sorted(CONFIGS), default="mad")
    p.add_argument("--cache", default=os.environ["SCRATCH"] + "/tvm_pool.pkl")
    p.add_argument("--pool", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--backend", choices=["tiled", "mixed"], default="tiled")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--compilation-cache", default="")
    p.add_argument("--sr-only", action="store_true")
    p.add_argument("--bm", type=int, default=0)
    p.add_argument("--bk", type=int, default=0)
    args = p.parse_args()
    select_config(args.config)
    global BM, BK
    if getattr(args, "bm", 0):
        BM = args.bm
    if getattr(args, "bk", 0):
        BK = args.bk
    (prep if args.mode == "prep" else trial)(args)


if __name__ == "__main__":
    main()
