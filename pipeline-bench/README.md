# PET input-pipeline benchmark harness

Snakemake matrix over the stacked metatrain branches in
[#1275](https://github.com/metatensor/metatrain/pull/1275). Lives here,
not in metatrain, so the PR branches stay mergeable.

Each cell is one `benchmarks/benchmark_pipeline.py` run against a
detached git worktree of that variant. Cells that need a GPU take
`gpu=1`, so a single GPU serializes them. That is the fair comparison;
do not run two cells on the same card.

Does not touch the shared `.venv`. Snakemake/pandas/matplotlib go in
`pipeline-bench/.venv`; the PET run uses `/home/ericb/metawork/.venv`.

## Setup

```bash
cd pipeline-bench
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

Silicon 216 is generated on demand (`data/si_bulk_216.xyz`). The other
two datasets are metatrain test resources.

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
  --python /home/ericb/metawork/.venv/bin/python \
  --dataset /home/ericb/metawork/metatrain/tests/resources/qm9_reduced_100.xyz \
  --key U0 --num-workers 0 --batch-size 8 --device cpu --epochs 2 \
  --out /tmp/cell.json --log /tmp/cell.log
```

## Cluster (kuma)

One GPU job that runs the DAG sequentially on the allocated node:

```bash
/home/ericb/metawork/.venv/bin/python /home/ericb/metawork/etc/hpc_run.py \
  /home/ericb/metawork/etc/hpc-jobs/examples/pipeline-bench.yaml
# then, on a login node: sbatch the printed job.sbatch
```

`profiles/slurm` is there if you later want one sbatch per cell
(`snakemake --profile profiles/slurm`). Fair GPU numbers still want
one cell per card.

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
