# PET input-pipeline benchmark harness

Snakemake matrix over the stacked metatrain branches in
[#1275](https://github.com/metatensor/metatrain/pull/1275). Lives here,
not in metatrain, so the PR branches stay mergeable.

Each cell is one `benchmarks/benchmark_pipeline.py` run against a
detached git worktree of that variant. Cells that need a GPU take
`gpu=1`, so a single GPU serializes them. That is the fair comparison;
do not run two cells on the same card.

Does not touch the shared `.venv`. Snakemake/pandas/matplotlib go in
`pipeline-bench/.venv`; the PET run uses `/home/boittier/metawork/.venv`.

## Setup

```bash
cd pipeline-bench
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

Silicon 216 is generated on demand (`data/si_bulk_216.xyz`). `qm9` and
`carbon` are metatrain test resources. `omol_proxy` (`data/omol_proxy.xyz`)
is a synthetic size-diverse dataset (2-512 atoms/structure) generated
on demand, standing in for a large diverse dataset like OMol25 -- see
[Distributed / atoms-per-batch](#distributed--atoms-per-batch-not-part-of-the-main-matrix).

## Local

Smoke (qm9, `everything`, 2 epochs, workers=0). Either flag works;
`--config smoke=true` overwrites the YAML mapping, so the Snakefile
re-reads the `smoke:` block from `config.yaml`.

```bash
.venv/bin/snakemake --profile profiles/local --config smoke=true
.venv/bin/snakemake --profile profiles/local --config run=smoke
```

Full matrix on this machine:

```bash
.venv/bin/snakemake --profile profiles/local
```

A cell can be run without Snakemake:

```bash
.venv/bin/python scripts/run_cell.py \
  --worktree worktrees/everything \
  --python /home/boittier/metawork/.venv/bin/python \
  --dataset /home/boittier/metawork/metatrain/tests/resources/qm9_reduced_100.xyz \
  --key U0 --num-workers 0 --batch-size 8 --device cpu --epochs 2 \
  --out /tmp/cell.json --log /tmp/cell.log
```

## Cluster (kuma)

One GPU job that runs the DAG sequentially on the allocated node:

```bash
/home/boittier/metawork/.venv/bin/python /home/boittier/metawork/etc/hpc_run.py \
  /home/boittier/metawork/etc/hpc-jobs/examples/pipeline-bench.yaml
# then, on a login node: sbatch the printed job.sbatch
```

Kuma has no default Slurm partition (`etc/hpc-jobs/examples/pipeline-bench.yaml`
sets `partition: l40s`) and rejects `--gres=gpu:N` combined with `--nodes` --
`etc/hpc_run.py` renders `--gpus=N` without `--nodes` for GPU jobs instead;
see the comments in `render_sbatch()` if porting this to another cluster.

`profiles/slurm` is there if you later want one sbatch per cell
(`snakemake --profile profiles/slurm`). Fair GPU numbers still want
one cell per card.

Ready-to-submit sbatch templates for one-off checks (not part of the
Snakemake-driven matrix) are in `slurm/`:

| template | what |
| --- | --- |
| `gpu-smoke.sbatch` | runs the `smoke:` config on an actual GPU (`--config smoke=true` normally targets CPU) |
| `pet-xl-smoke.sbatch` | same, with a much larger PET model (`--model-hypers-json`) to check param count / GPU memory |
| `multigpu-smoke.sbatch` | real 2-GPU DDP via `srun`, ad hoc (outside the `distributed` Snakemake target) |
| `distributed-sweep.sbatch` | runs the formalized `distributed` Snakemake target (below) |
| `build-stack.sbatch` | rebuilds `metatensor[torch]`/`metatomic[torch]` on a compute node -- the login node caps every user's cgroup at 8GiB, not enough for parallel C++ compiles |

All of them `--chdir` into `pipeline-bench/` and log to `smoke-slurm/`
(gitignored, one-off run output -- not the same thing as `slurm/`, the
tracked templates).

## Variants

| name | refs | what it measures |
| --- | --- | --- |
| `baseline` | `origin/perf/1-pipeline-benchmark` | timing only |
| `persistent` | `origin/perf/4-persistent-workers` | #1268–#1271 |
| `timed` | `origin/pr/collate-transform-timing` | per-transform stages (#1274) |
| `pinned` | timed + `origin/pr/pin-batches` | pinning on the timed stack. Those two branches add kwargs at the same DataLoader sites, so the worktree merge keeps both sides of the add/add conflict. |
| `transport` | `origin/pr/system-tensor-transport` | tensor transport, not opened |
| `everything` | `origin/perf/5-batch-transport` | #1275, the integration ref |

The worktree's `src/` is put on `PYTHONPATH` so the shared venv's
editable metatrain install does not shadow the variant. `src/metatrain/_version.py`
is generated at install time and is not in git, so the worktree copies it from
the live checkout.

## Outputs

- `results/cells/<variant>/<dataset>/w*_b*_<device>_r*.json` — one cell
- `results/cells.csv`, `results/stages.csv` — tidy tables
- `results/summary.md` — median over repeats, speedup vs `baseline`
- `results/atoms_per_s.png` — optional, needs matplotlib

## Distributed / atoms-per-batch (not part of the main matrix)

A separate `distributed:` block in `config.yaml` sweeps `world_size` (real
multi-GPU `DistributedDataParallel`, via `srun --ntasks`) and
`max_atoms_per_batch` (packs batches by total atom count instead of a fixed
structure count) on the `omol_proxy` dataset. It is opt-in and separate from
`rule all` on purpose: unlike every other cell in this harness, a
`world_size > 1` cell needs more than one GPU, so it needs a different Slurm
allocation than the rest of the matrix (see `slurm/distributed-sweep.sbatch`,
which requests `--ntasks=2 --gpus-per-node=2` to match the default
`world_sizes: [1, 2]`).

```bash
sbatch --partition=l40s --job-name=distributed-sweep slurm/distributed-sweep.sbatch
```

`metatrain`'s `Trainer.train()` needed no changes for this: it auto-enables
`DistributedDataParallel` whenever `SLURM_NTASKS > 1`
(`metatrain.utils.distributed.slurm.resolve_distributed`) and does its own
process-group setup from the Slurm environment. `run_cell.py --world-size N`
just launches the identical benchmark command via `srun --ntasks N`, one
rank per GPU, and reads back rank 0's log (each rank's stdout goes to its own
file -- interleaving N ranks into one pipe would garble it). `batch_size` in
`distributed:` is per-rank; global batch = `batch_size * world_size`.
`max_atoms_per_batch` uses metatrain's existing `MaxAtomBatchSampler` /
`MaxAtomDistributedBatchSampler` (`training.max_atoms_per_batch` hyper) --
no new sampler code, just exposing an existing hyper as a CLI flag.

Measured on `omol_proxy`, 2xL40S, `max_atoms_per_batch=512`: 2-GPU DDP gave
~1.3x throughput per GPU-pair, not 2x -- `backward` time roughly doubled
(17.7ms to 35.7ms/call) from NCCL gradient all-reduce, not amortized at this
dataset's scale (~20 steps/epoch/rank). For a real OMol-scale run, per-GPU
batch (in atoms, not structures) needs to be large enough that compute time
dominates communication time, or most of the wall-clock goes to
synchronizing gradients rather than training.

Outputs mirror the main matrix: `results/distributed/<variant>/g<world_size>_r<repeat>.json`,
`results/distributed_cells.csv`, `results/distributed_stages.csv`,
`results/distributed_summary.md`.
