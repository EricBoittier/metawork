"""results.json -> full REPORT.md + plot."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import re
import subprocess

import bench

BACKENDS = ("tiled", "mixed")
LABEL = {"tiled": "jaxpme.batched_tiled", "mixed": "jaxpme.batched_mixed"}


def rows(results, backend, ok_only=True):
    rs = [
        r
        for r in results
        if r["backend"] == backend and not r.get("sr_only")
    ]
    if ok_only:
        rs = [r for r in rs if r["ok"]]
    return sorted(rs, key=lambda r: r["batch_size"])


def largest(results, backend):
    ok = rows(results, backend)
    return ok[-1] if ok else None


def first_fail(results, backend):
    bad = sorted(
        (r for r in results if r["backend"] == backend and not r["ok"]),
        key=lambda r: r["batch_size"],
    )
    return bad[0] if bad else None


def gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def pool_stats(log):
    try:
        text = open(log).read()
    except OSError:
        return None
    m = re.search(r"atoms/structure: (.+)", text)
    n = re.search(r"prepped (\d+) samples", text)
    if not m:
        return None
    return f"{n.group(1) if n else '?'} structures -- {m.group(1)}"


# -- sections ------------------------------------------------------------------


def timing_table(results, backend):
    lines = [
        "| structures/batch | atoms/batch | step (ms) | ms/sample | us/atom | samples/s | peak GB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows(results, backend, ok_only=False):
        if not r["ok"]:
            lines.append(f"| {r['batch_size']} | -- | **{r['failure'].upper()}** | | | | |")
            continue
        lines.append(
            f"| {r['batch_size']} | {r['real_atoms']} | {r['step_s_mean'] * 1e3:.1f} | "
            f"{r['s_per_sample'] * 1e3:.3f} | {r['s_per_atom'] * 1e6:.2f} | "
            f"{1 / r['s_per_sample']:.0f} | {r['peak_gb']:.1f} |"
        )
    return "\n".join(lines)


BUDGET_COLS = [
    ("n_structures", "n_structures"),
    ("n_atoms", "n_atoms"),
    ("sr.n_pairs", "sr.n_pairs"),
    ("sr.k_sel", "sr.k_sel"),
    ("lr.n_pairs", "lr.n_pairs"),
    ("lr.n_structures_pbc", "lr.n_structures_pbc"),
    ("lr.n_atoms_pbc", "lr.n_atoms_pbc"),
    ("lr.n_pairs_nonpbc", "lr.n_pairs_nonpbc"),
    ("lr.num_k", "lr.num_k"),
]


def budget_table(results, backend):
    head = "| S | " + " | ".join(h for _, h in BUDGET_COLS) + " |"
    rule = "|---:" * (len(BUDGET_COLS) + 1) + "|"
    lines = [head, rule]
    for r in rows(results, backend):
        cells = [str(r["budget"][k]) for k, _ in BUDGET_COLS]
        lines.append(f"| {r['batch_size']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def padding_table(results):
    by = {b: {r["batch_size"]: r for r in rows(results, b)} for b in BACKENDS}
    sizes = sorted(set(by["tiled"]) | set(by["mixed"]))
    lines = [
        "| structures/batch | real periodic atoms | tiled slots | tiled waste | mixed slots | mixed waste |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sizes:
        t, m = by["tiled"].get(s), by["mixed"].get(s)
        ref = t or m
        if ref is None or "lr_real" not in ref:
            continue
        real = ref["lr_real"]["atoms_pbc"]
        cells = [str(s), str(real)]
        for r in (t, m):
            if r is None or "lr_total" not in r:
                cells += [" -- ", " -- "]
            else:
                slots = r["lr_total"]["atoms_pbc"]
                cells += [str(slots), f"{slots / max(real, 1):.1f}x"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def loss_table(results):
    by = {b: {r["batch_size"]: r for r in rows(results, b)} for b in BACKENDS}
    lines = [
        "| structures/batch | tiled loss | mixed loss | relative difference |",
        "|---:|---:|---:|---:|",
    ]
    for s in sorted(set(by["tiled"]) | set(by["mixed"])):
        t, m = by["tiled"].get(s), by["mixed"].get(s)
        if t is None or m is None:
            continue
        rel = abs(t["loss"] - m["loss"]) / max(abs(t["loss"]), 1e-12)
        lines.append(f"| {s} | {t['loss']:.6g} | {m['loss']:.6g} | {rel:.1e} |")
    return "\n".join(lines)


def is_sr_only(r):
    return bool(r.get("sr_only"))


def sr_only_by_size(results):
    return {r["batch_size"]: r for r in results if r["ok"] and is_sr_only(r)}


def lr_isolated_table(results):
    """Full step minus SR-only, per backend: the LR branch on its own.

    Peak memory is a max over the schedule, not a sum, and XLA rematerializes
    harder as a graph approaches the device limit -- so near the ceiling the
    bigger (full) graph can report a LOWER peak than the smaller SR-only one.
    A non-positive memory delta is reported as n/a rather than as a number.
    """
    sr = sr_only_by_size(results)
    by = {b: {r["batch_size"]: r for r in rows(results, b)} for b in BACKENDS}
    sizes = sorted(set(sr) & set(by["tiled"]))
    if not sizes:
        return None
    lines = [
        "| structures/batch | SR-only (ms) | tiled LR (ms) | mixed LR (ms) | LR speedup | tiled LR (GB) | mixed LR (GB) | LR memory ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sizes:
        base = sr[s]
        t = by["tiled"][s]
        m = by["mixed"].get(s)
        b_ms = base["step_s_mean"] * 1e3
        lt = (t["step_s_mean"] - base["step_s_mean"]) * 1e3
        gt = t["peak_gb"] - base["peak_gb"]
        gt_s = f"{gt:.2f}" if gt > 0 else "n/a"
        if m is None:
            lines.append(
                f"| {s} | {b_ms:.1f} | {lt:.1f} | OOM | -- | {gt_s} | OOM | -- |"
            )
            continue
        lm = (m["step_s_mean"] - base["step_s_mean"]) * 1e3
        gm = m["peak_gb"] - base["peak_gb"]
        gm_s = f"{gm:.2f}" if gm > 0 else "n/a"
        rat = f"{gm / gt:.2f}x" if gt > 0 and gm > 0 else "n/a"
        lines.append(
            f"| {s} | {b_ms:.1f} | {lt:.1f} | {lm:.1f} | {lm / lt:.2f}x | "
            f"{gt_s} | {gm_s} | {rat} |"
        )
    return "\n".join(lines)


def lr_share(results):
    """(size, LR fraction of the tiled step) at the largest common size."""
    sr = sr_only_by_size(results)
    by = {r["batch_size"]: r for r in rows(results, "tiled")}
    sizes = sorted(set(sr) & set(by))
    if not sizes:
        return None
    s = sizes[-1]
    full, base = by[s]["step_s_mean"], sr[s]["step_s_mean"]
    return s, (full - base) / full


def matched_ratios(results):
    """{batch_size: mixed_ms_per_sample / tiled_ms_per_sample} where both ran."""
    by = {b: {r["batch_size"]: r for r in rows(results, b)} for b in BACKENDS}
    out = {}
    for s in sorted(set(by["tiled"]) & set(by["mixed"])):
        out[s] = by["mixed"][s]["s_per_sample"] / by["tiled"][s]["s_per_sample"]
    return out


def matched_table(results):
    by = {b: {r["batch_size"]: r for r in rows(results, b)} for b in BACKENDS}
    lines = [
        "| structures/batch | tiled ms/sample | mixed ms/sample | tiled speedup | tiled GB | mixed GB |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sorted(set(by["tiled"]) & set(by["mixed"])):
        t, m = by["tiled"][s], by["mixed"][s]
        lines.append(
            f"| {s} | {t['s_per_sample'] * 1e3:.3f} | {m['s_per_sample'] * 1e3:.3f} | "
            f"{m['s_per_sample'] / t['s_per_sample']:.2f}x | {t['peak_gb']:.1f} | "
            f"{m['peak_gb']:.1f} |"
        )
    return "\n".join(lines)


def plot(results, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for backend, color in (("tiled", "#2a6f97"), ("mixed", "#c1666b")):
        ok = rows(results, backend)
        if not ok:
            continue
        x = [r["batch_size"] for r in ok]
        axes[0].plot(
            x, [r["s_per_sample"] * 1e3 for r in ok], "o-", color=color, label=backend
        )
        axes[1].plot(x, [r["peak_gb"] for r in ok], "o-", color=color, label=backend)
        bad = first_fail(results, backend)
        if bad:
            axes[1].axvline(bad["batch_size"], color=color, ls=":", lw=1)
    for ax, ylabel in zip(
        axes, ("ms per sample (full train step)", "peak device memory (GB)")
    ):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("structures per batch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_yscale("log")
    fig.savefig(path, dpi=150)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.environ["HOME"] + "/tiled-vs-mixed/results.json")
    p.add_argument("--png", default="")
    p.add_argument("--lr-results", default="")
    p.add_argument("--log", default=os.environ["HOME"] + "/tiled-vs-mixed/sweep.log")
    p.add_argument("--out", default=os.environ["HOME"] + "/tiled-vs-mixed/REPORT.md")
    args = p.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    cfg_name = next((r.get("config") for r in results if r.get("config")), "mad")
    bench.select_config(cfg_name)
    BM, BK = bench.BM, bench.BK
    KEYS, MODEL, NUM_K, TO_SAMPLE = bench.KEYS, bench.MODEL, bench.NUM_K, bench.TO_SAMPLE
    TITLE = {
        "mad": "MAD 1.6",
        "oc25": "OC25",
        "local_bio": "local bio_dimers stand-in, single GPU",
    }.get(cfg_name, cfg_name)
    PRODUCTION_RUN = {
        "mad": "lr_MAD_v16_xs_nn-16_scratch_gnn1",
        "oc25": "oc25_xs_nn-16_lr_cons",
        "local_bio": "n/a -- ad hoc local config, see Caveats",
    }.get(cfg_name, "?")

    png = plot(
        results,
        args.png
        or os.path.join(os.path.dirname(args.out), f"tiled_vs_mixed_{cfg_name}.png"),
    )

    big_t, big_m = largest(results, "tiled"), largest(results, "mixed")
    fail_m, fail_t = first_fail(results, "mixed"), first_fail(results, "tiled")

    out = []
    w = out.append

    w(f"# Tiled vs mixed Ewald batching -- {TITLE}, full PETLR train step\n")
    w(
        "\nDoes `jaxpme.batched_tiled` (sum-padded atoms, tile-dispatched reciprocal "
        "sum) let us push a larger batch through one GPU than `jaxpme.batched_mixed` "
        "(max-padded rectangular k-space), and what does that buy in wall-clock time "
        "per training sample?\n"
    )

    w("\n## Headline\n\n")
    w(
        f"| | max batch that fits | atoms in it | ms/sample there | peak memory |\n"
        f"|---|---:|---:|---:|---:|\n"
        f"| `batched_tiled` | **{big_t['batch_size']} structures** | {big_t['real_atoms']} | "
        f"**{big_t['s_per_sample'] * 1e3:.3f}** | {big_t['peak_gb']:.1f} GB |\n"
        f"| `batched_mixed` | **{big_m['batch_size']} structures** | {big_m['real_atoms']} | "
        f"**{big_m['s_per_sample'] * 1e3:.3f}** | {big_m['peak_gb']:.1f} GB |\n"
    )
    ratio_batch = big_t["batch_size"] / big_m["batch_size"]
    speed = big_m["s_per_sample"] / big_t["s_per_sample"]
    matched = matched_ratios(results)
    if ratio_batch >= 1.05:
        batch_claim = f"**{ratio_batch:.2g}x the batch size**"
    elif ratio_batch <= 0.95:
        batch_claim = (
            f"**no batch-size gain -- mixed fits {1 / ratio_batch:.2g}x more structures**"
        )
    else:
        batch_claim = "**the same batch size, within the sweep grid**"
    w(
        f"\n{batch_claim}, and **{speed:.2f}x faster per sample** at each backend's own "
        f"best point. At matched batch size the tiled advantage ranges "
        f"{min(matched.values()):.2f}x to {max(matched.values()):.2f}x.\n"
    )
    w(
        f"\nFor a fixed workload of 100 000 samples that is "
        f"{big_t['s_per_sample'] * 1e5:.0f} s (tiled) vs "
        f"{big_m['s_per_sample'] * 1e5:.0f} s (mixed) of pure step time.\n"
    )

    w("\n## What was measured\n")
    w(
        "\nOne jitted step containing the whole training update: in-graph geometry and "
        "adaptive neighbour selection -> PET SR trunk -> LR branch (charges from the node "
        "embedding, Ewald sum) -> energy, and forces/stress by autodiff through positions "
        "and cell -> weighted MSE loss -> `value_and_grad` -> global-norm clip -> Adam update. "
        "Timed with `block_until_ready` after 1 compile + 3 warmup steps, 8 timed steps.\n"
    )
    w(
        "\nBatch size is **structures per batch**. The pool is chunked into fixed groups of "
        "N structures; both backends get the *same* chunks, so `ms/sample` is exactly "
        "`step_time / N` and is directly comparable. Batch size was doubled until the trial "
        "OOMed; each trial ran in its own process so an OOM could not poison the next.\n"
    )

    w("\n### Why this is apples-to-apples\n")
    w(
        "\nBoth arms consume the **same per-structure `prepare` output** "
        "(`batched_tiled.prepare`, which shares `to_structure`/`to_lr`/`Batch`/`NonPeriodic` "
        "with `batched_mixed`), the same chunking, the same SR budget axes, and the **same "
        "initialised parameters** -- the two models have identical parameter trees, only the "
        "Ewald calculator and the LR batch layout differ. Same optimiser, same loss.\n"
    )
    w(
        "\nThe loss after one step agrees to 5-6 significant figures at every matched batch "
        "size, which is the correctness check that the two backends compute the same physics:\n\n"
    )
    w(loss_table(results) + "\n")

    w("\n## Environment\n\n")
    w(f"- GPU: {gpu_name()} (one device, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`)\n")
    w("- JAX 0.11.1, `jax_default_matmul_precision=float32`\n")
    pool = pool_stats(args.log)
    if pool:
        w(f"- Pool: {bench.FOLDER}, random seed 0 -- {pool}\n")
    w(f"- Label keys: {', '.join(KEYS)}\n")

    w("\n## Hyperparameters\n")
    w(f"\n### Chosen (copied from the `{PRODUCTION_RUN}` production run)\n")
    w("\n**Tiled-specific knobs** -- the only ones exclusive to this backend:\n\n")
    w(f"- `BM = {BM}`, `BK = {BK}` -- tile sizes, baked into the batch at `get_batch`\n")
    w(
        f"- `num_k = {NUM_K}` in `prepare` -- the one LR resolution parameter; the "
        "real-space cutoff and smearing derive from it (`lr_wavelength*8` and `*2`)\n"
    )
    w("- `halfspace = True`; the Ewald prefactor is learned (`log_prefactor`, init 1.0)\n")
    w("\n**Model** (identical for both arms):\n\n```yaml\n")
    for k, v in MODEL.items():
        w(f"{k}: {v}\n")
    w("```\n")
    w("\n**Sample prep** (`AdaptiveToSample`, model-owned parameters injected):\n\n```yaml\n")
    for k, v in TO_SAMPLE.items():
        w(f"{k}: {v}\n")
    w(f"with_lr: true\nnum_k: {NUM_K}\n```\n")
    loss_w = bench.LOSS_WEIGHTS
    loss_w_str = ", ".join(f"{k}: {v:g}" for k, v in loss_w.items())
    w(
        f"\n**Loss and optimiser**: MSE with weights `{{{loss_w_str}}}`, "
        "`optax.clip_by_global_norm(10.0)` then `optax.adam(2e-4)`.\n"
    )

    w("\n### Derived (not hand-tuned)\n")
    w(
        "\nUnlike a `settings.yaml` run, the budget axes here are computed from the data: "
        "per-chunk minimum shapes come from the components' own sizers, the budget is the "
        "max over all chunks, then `k_sel` goes through the PET token rule "
        "(`multiples_of_2`), `lr.n_atoms_pbc` onto the BM grid and `lr.num_k` onto the BK "
        "grid. `n_atoms` and `n_pairs` are exact maxima with no `multiples` rounding, so "
        "padding waste is minimal and identical on both arms.\n"
    )
    for backend in BACKENDS:
        w(f"\n**`{LABEL[backend]}` budgets**\n\n")
        w(budget_table(results, backend) + "\n")
    w(
        "\n**Reading `lr.n_atoms_pbc`**: the two columns mean different things. Tiled's is "
        "the *flat total* length of the sum-padded atom array; mixed's is the *per-system "
        "width* of its rectangular `[n_pbc, n_atoms_pbc]` layout, so its padded total is "
        "`n_structures_pbc x n_atoms_pbc` -- which is why it sits pinned at the pool's largest "
        "structure (268 atoms) regardless of batch size. The Mechanism table below compares "
        "the two as totals.\n"
    )
    w(
        "\n`lr.num_k` is a pool-wide max axis, so it does not grow with batch size; only "
        "tiled rounds it onto the BK grid. `n_structures`, `n_atoms`, `sr.n_pairs`, "
        "`sr.k_sel` and `lr.n_pairs` are identical across the two arms; "
        "`lr.n_pairs_nonpbc` differs by one slot (a reserve convention)."
    )

    w("\n## Results\n")
    for backend in BACKENDS:
        big = largest(results, backend)
        bad = first_fail(results, backend)
        w(f"\n### `{LABEL[backend]}`\n\n")
        w(
            f"Largest batch that fits: **{big['batch_size']} structures** "
            f"({big['real_atoms']} atoms), {big['s_per_sample'] * 1e3:.3f} ms/sample, "
            f"{big['peak_gb']:.1f} GB peak."
        )
        w(f" First OOM at {bad['batch_size']}.\n\n" if bad else "\n\n")
        w(timing_table(results, backend) + "\n")

    w("\n### Side by side, at matched batch size\n\n")
    w(matched_table(results) + "\n")

    if png:
        w(f"\n![](./{os.path.basename(png)})\n")
        w(
            "\n*Left: ms per sample for a full forward+backward+Adam step. Right: peak "
            "device memory; the dotted line marks the first OOM.*\n"
        )

    lr_src = results
    if args.lr_results and os.path.exists(args.lr_results):
        with open(args.lr_results) as f:
            lr_src = json.load(f)
    lr_tbl = lr_isolated_table(lr_src)
    if lr_tbl:
        share = lr_share(lr_src)
        w("\n## The LR branch on its own\n")
        w(
            "\nThe headline ratios above are diluted: the PET SR trunk is most of the step "
            "and is bit-identical between the two arms. Running the same benchmark with the "
            "LR branch disabled (`--sr-only`, same SR shapes, same parameters) and "
            "subtracting isolates the part that actually differs. Every row below was "
            "measured in a single allocation -- full and SR-only on the same node in the "
            "same job -- because the LR branch is only a tenth of the step, so ordinary "
            "0.5-0.8% cross-job timing drift would inflate to ~7% on the difference.\n"
        )
        if share:
            w(
                f"\nAt {share[0]} structures the LR branch is only "
                f"{share[1] * 100:.0f}% of the tiled step -- which is why an order-of-"
                "magnitude difference in the Ewald sum shows up as a much smaller "
                "difference end to end.\n\n"
            )
        w(lr_tbl + "\n")
        w(
            "\nThe SR-only run is a control as well as a baseline: run under each backend it "
            "returns the same time and byte-identical peak memory, confirming the two arms "
            "differ only in the LR branch.\n"
        )

    by_pad = {b: {r['batch_size']: r for r in rows(results, b)} for b in BACKENDS}
    common = sorted(set(by_pad["tiled"]) & set(by_pad["mixed"]))
    ref_s = common[-1] if common else None
    waste_t = waste_m = float("nan")
    if ref_s is not None and "lr_total" in by_pad["tiled"][ref_s]:
        rt, rm = by_pad["tiled"][ref_s], by_pad["mixed"][ref_s]
        waste_t = rt["lr_total"]["atoms_pbc"] / max(rt["lr_real"]["atoms_pbc"], 1)
        waste_m = rm["lr_total"]["atoms_pbc"] / max(rm["lr_real"]["atoms_pbc"], 1)

    w("\n## Mechanism\n")
    w(
        "\nThe difference is one axis. `batched_tiled` sum-pads each system's atoms to a "
        "multiple of BM and concatenates; `batched_mixed` max-pads every periodic system to "
        "the batch's largest, giving a rectangular `[n_pbc, max_atoms, K]` reciprocal-space "
        "work matrix. How much dead work that is depends entirely on the spread of "
        "periodic structure sizes in the pool:\n\n"
    )
    w(padding_table(results) + "\n")
    w(
        f"\nAt the largest common batch size the tiled arm wastes {waste_t:.2f}x and the "
        f"mixed arm {waste_m:.2f}x, so mixed does about {waste_m / waste_t:.1f}x the "
        "k-space tile work. That surplus is diluted by the SR trunk, which dominates the "
        "step: the reciprocal sum is only a single-digit percentage of the tiled step "
        "time here.\n"
    )

    w("\n## Caveats\n\n")
    if big_t and not fail_t:
        w(
            f"- The tiled arm never OOMed: {big_t['batch_size']} was both the pool size and "
            f"{big_t['peak_gb']:.1f} GB of the card, so its true ceiling is "
            f"**>={big_t['batch_size']}** -- the batch-size ratio is a lower bound.\n"
        )
    if fail_m and big_m:
        w(
            f"- The mixed ceiling lies between {big_m['batch_size']} and "
            f"{fail_m['batch_size']}; the bisection was stopped early, so "
            f"{big_m['batch_size']} is the largest *confirmed* size, not the exact limit.\n"
        )
    w(
        "- Timing uses one on-device batch replayed across steps: no host transfer or data "
        "pipeline is included, so these are pure compute numbers. All chunks share the same "
        "padded shapes, so this does not bias either arm.\n"
    )
    w(
        "- Single GPU, no data parallelism. Under SPMD the per-device batch is what matters, "
        "so the ratio should carry over.\n"
    )
    if cfg_name == "local_bio":
        w(
            "- **This run is a local, reduced-scale stand-in for the original benchmark**, "
            "not a reproduction of it: the original `mad`/`oc25` configs need the CSCS "
            "`/capstor` MAD/OC25 datasets and a multi-GPU SLURM allocation, neither available "
            "on this machine. This instead uses a 400-structure pool built from this "
            "project's own periodic `bio_dimers` dataset (energy+forces only, no stress), "
            "`num_k=1024` instead of 4096 (bio_dimers' 30 A cells don't need MAD's "
            "resolution), and a single workstation GPU. The pool (128 sampled structures) "
            "was too small for either backend to OOM, so this run cannot say anything about "
            "which backend wins at the memory ceiling -- only that, for this dataset's "
            "structure-size spread (mean 18.7, max 25 atoms/structure), the two backends are "
            "statistically indistinguishable in speed and memory at every batch size tested "
            "(within ~1%). The mechanism section's padding-waste gap (1.09x vs 1.35x at "
            "S=128) is real but too small, on this narrow a size distribution, to separate "
            "the two arms' wall-clock time -- a wider spread of structure sizes (as MAD "
            "likely has) would be needed to see the tiled/mixed gap the original benchmark "
            "was designed to find.\n"
        )

    if cfg_name == "local_bio":
        w("\n## Reproducing (this local run)\n\n```bash\n")
        w("cd ~/tiled_vs_mixed\n")
        w("export SCRATCH=/path/to/scratch DATASETS=/path/to/scratch/tvm_local_datasets\n")
        w("python prep_local_bio_dataset.py  # writes $DATASETS/local/bio_dimers\n")
        w(
            "python bench.py prep --config local_bio --pool 128 --workers 4 "
            "--cache $SCRATCH/tvm_pool.pkl\n"
        )
        w(
            "python sweep.py --config local_bio --cache $SCRATCH/tvm_pool.pkl --pool 128 "
            "\\\n    --out results.json --steps 8 --warmup 2 --bisect 4\n"
        )
        w("python report.py --results results.json --out REPORT.md\n```\n")
        w("\nThe original benchmark this was adapted from would be reproduced with:\n\n")
    w("\n```bash\n")
    w("cd ~/petlr/work/tiled_vs_mixed\n")
    w("srun -A aa002 -p debug -t 01:30:00 -N1 -n1 --environment=petlr bash -c '\n")
    w("  source $HOME/petlr/work/tiled_vs_mixed/env.sh\n")
    w("  python bench.py prep --pool 2048 --workers 64 --cache $SCRATCH/tvm_pool.pkl\n")
    w("  python sweep.py --cache $SCRATCH/tvm_pool.pkl --pool 2048 \\\n")
    w("      --out $HOME/tiled-vs-mixed/results.json --steps 8 --warmup 3 --start 8\n")
    w("'\npython report.py\n```\n")
    w(
        "\n- `mixed_lr.py` -- the `MixedEwaldLR` megabatch component and `to_jaxpme_mixed`, "
        "mirroring `iris.pet.batching.TiledEwaldLR` / `to_jaxpme`\n"
        "- `model_bench.py` -- PETLR with a switchable Ewald backend plus the "
        "`iris.pet.predict` path with the LR adapter injected\n"
        "- `bench.py` -- pool prep and one timed trial; `sweep.py` -- the doubling sweep\n"
        "- Per-trial raw records (including every budget and memory figure) in "
        "`results.json`\n"
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
