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

Silicon 216 is generated on demand (`data/si_bulk_216.xyz`). `si_large`
(1000 atoms/structure, 128 structures) isn't wired into a Snakemake
generation rule yet — regenerate it by hand if `data/si_bulk_1000_128.xyz`
is missing:

```bash
.venv/bin/python scripts/make_si_bulk.py \
  --out data/si_bulk_1000_128.xyz --n-cells 5 --n-structures 128 --seed 1
```

The other two datasets are metatrain test resources.

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

`baseline`'s `benchmark_pipeline.py` validated on the first 8 structures of
the training set itself — not a disjoint split, and its header line didn't
match the `"{n} train + {m} validation structures"` format the later
branches use, so `results/cells.csv` silently had no `n_train`/`n_val` for
it. Fixed on `fix/pipeline-benchmark-val-split` (branched off
`perf/1-pipeline-benchmark`) — merge that into `baseline`'s ref before
trusting this harness's `results/equivalence.md` for it.

## Outputs

- `results/cells/<variant>/<dataset>/w*_b*_<device>_r*.json` — one cell
- `results/cells.csv`, `results/stages.csv` — tidy tables
- `results/summary.md` — median over repeats, speedup vs `baseline`, plus a
  spread column and the methodology caveats below
- `results/equivalence.md` — do variants sharing a dataset agree on
  `n_train`/`n_val`? Catches a workload that silently isn't the same across
  branches (e.g. a non-disjoint validation split). Does **not** check model
  correctness — no loss/gradient/output comparison exists between variants
  anywhere in this harness; that's open follow-up work, not something this
  script does.
- `results/drift.md` — does measured speed correlate with chronological run
  order, globally or within a variant's own block? Snakemake doesn't
  randomize job order against variant identity, so a multi-minute run's
  thermal/load drift would otherwise be silently confounded with whichever
  variant happened to run early or late. A flagged rho is a smell test, not
  a diagnosis — see the script's own caveat in its output.
- `results/atoms_per_s.png` — optional, needs matplotlib

## Methodology gaps this harness cannot close by itself

Fixed here: single-batch-size testing, low repeat count, no order/thermal
check, no cross-variant workload-shape check. Still open, and out of scope
for a pipeline-bench-only change:

- **No correctness check on model outputs.** Every variant is compared on
  wall-clock speed alone. A change that's faster but silently wrong (drops
  a structure, corrupts a batch, races on a shared buffer) would look
  identical to a real win here. Closing this needs each branch's
  `benchmark_pipeline.py` to report something like a final loss value that
  can be compared across variants, not a pipeline-bench change.
- **Warm-up isn't excluded from the timed region.** More epochs (see
  `config.yaml`) dilutes it but doesn't remove it; excluding it needs
  per-epoch timing in `benchmark_pipeline.py`, which doesn't exist yet.
