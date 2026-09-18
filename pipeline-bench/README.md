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
the training set itself — not a disjoint split, and used a fixed 8-structure
val set instead of the `val_fraction=0.2` every later branch defaults to, so
its `n_train`/`n_val` didn't match the others' even once disjoint. Both
fixed on `fix/pipeline-benchmark-val-split` (branched off
`perf/1-pipeline-benchmark`) — merge that into `baseline`'s ref before
trusting this harness's `results/equivalence.md` for it.

Every variant's `benchmark_pipeline.py` also now seeds `random`/`numpy`/
`torch` with the same fixed `SEED = 0` before touching the dataset or model,
and prints `trainer.best_metric` (the best validation metric `Trainer.train`
tracks internally, independent of `log_interval`) as a `best_val_metric`
line. That's what `results/correctness.md` compares across variants — see
below. One branch per variant's own real ref, same instrumentation on each:

| variant | ref | fix branch |
| --- | --- | --- |
| `persistent` | `perf/4-persistent-workers` | `fix/report-best-metric` |
| `timed` | `pr/collate-transform-timing` | `fix/report-best-metric-timed` |
| `transport` | `pr/system-tensor-transport` | `fix/report-best-metric-transport` |
| `everything` | `perf/5-batch-transport` | `fix/report-best-metric-everything` |

(Distinct names because each is branched off a different, divergent ref —
a shared name would collide on push.) `pinned` needs no separate fix since
it's built from `timed`'s ref plus `pr/pin-batches`, and will pick this up
on its next worktree rebuild once `fix/report-best-metric-timed` lands.

## Outputs

- `results/cells/<variant>/<dataset>/w*_b*_<device>_r*.json` — one cell
- `results/cells.csv`, `results/stages.csv` — tidy tables
- `results/summary.md` — median over repeats, speedup vs `baseline`, plus a
  spread column and the methodology caveats below
- `results/equivalence.md` — do variants sharing a dataset agree on
  `n_train`/`n_val`? Catches a workload that silently isn't the same across
  branches (e.g. a non-disjoint validation split). Checks workload *shape*
  only, not model correctness — see `results/correctness.md` for that.
- `results/correctness.md` — do variants agree on `best_val_metric`,
  epoch-1 training loss, and epoch-1 validation loss for the same config
  and seed, and does each variant reproduce itself across repeats? This is
  the one check in the harness that looks at whether variants compute the
  same thing rather than how fast they do it. `--config correctness=true`
  runs it across a few fixed seeds (see `config.yaml`'s `correctness:`
  block) so a WARN/FAIL isn't a coincidence of one seed. WARN at >=5%
  relative difference from `baseline`, FAIL at >=50% — **not** gated on
  `rule all` (see caveat below on why). Three scalars from a handful of
  seeds over a few epochs on tiny data, not a proof of numerical
  equivalence at the loss/gradient level, but a real signal — and a real
  one found immediately: see below.
- `results/drift.md` — does measured speed correlate with chronological run
  order, globally or within a variant's own block? Snakemake doesn't
  randomize job order against variant identity, so a multi-minute run's
  thermal/load drift would otherwise be silently confounded with whichever
  variant happened to run early or late. A flagged rho is a smell test, not
  a diagnosis — see the script's own caveat in its output.
- `results/atoms_per_s.png` — optional, needs matplotlib

## What the correctness check already found

Ran `--config correctness=true` for real (72 cells: 6 variants x 4 datasets
x 3 seeds, workers=0, batch=8, epochs=6) against the fix branches above.
`persistent`/`timed`/`pinned`/`transport`/`everything` agree with each other
bit-for-bit at every seed and every dataset — no disagreement among the five
newer branches, anywhere. `baseline` is a different story, and the size
pattern is stark (`results/correctness.md`'s seed-stability rollup, relative
difference on `best_val_metric` vs. the other five, min-max across 3 seeds):

| dataset | atoms/structure | baseline's gap across seeds |
| --- | ---: | --- |
| `si_large` | 1000 | 0%-0% |
| `si216` | 216 | 0%-1% |
| `carbon` | 4 | 6%-21% |
| `qm9` | ~10 | 8%-27% |

Baseline matches the rest of the stack almost exactly on the two bulk
datasets and diverges substantially and consistently on the two tiny
molecular datasets — the same structure-size axis that decided the
throughput story in `notes/medium-hardware-run.md`, but here it's about
whether the numbers agree at all, not how fast they're computed. One cell
(`qm9`, seed 0, `epoch1_val_loss`) crossed the 50% FAIL line outright (57%).
Plausible mechanism: with only ~10 atoms/structure, a batch carries very
little signal, so whatever incidental difference exists between baseline's
DataLoader construction and the newer branches' (different code, matched
seed, but not necessarily the same sequence of random draws once workers or
collate differ) gets amplified in the loss; with 1000 atoms/structure, each
batch carries enough signal that the same incidental difference washes out.
Not confirmed — a real explanation needs tracing where baseline's RNG
consumption actually diverges, which is open.

## Methodology gaps this harness cannot close by itself

Fixed here: single-batch-size testing, low repeat count, no order/thermal
check, no cross-variant workload-shape check, no correctness check at all,
single-seed correctness testing, correctness check limited to a derived
best-epoch metric. Still open:

- **Warm-up isn't excluded from the timed region.** More epochs (see
  `config.yaml`) dilutes it but doesn't remove it; excluding it needs
  per-epoch timing in `benchmark_pipeline.py`, which doesn't exist yet.
- **Still not a gradient-level equivalence test.** `best_val_metric` and
  epoch-1 train/val loss are real signals — reading `Trainer.train`'s own
  internal state, not reimplemented — but they're still scalars a few steps
  into training, not a tensor-level comparison of gradients or model
  outputs. Two variants could plausibly agree on all three and still differ
  somewhere those scalars don't reach.
- **`check_correctness.py`/`check_equivalence.py` don't gate `rule all`
  anymore.** They did originally (`sys.exit(1)` on a FAIL), until that was
  caught failing exactly when it mattered: Snakemake deletes a rule's
  output the moment its shell command exits non-zero, so the first real
  FAIL this harness hit deleted its own diagnostic report. Both scripts now
  always exit 0 and print the FAIL count to stderr — read the `.md` file,
  don't rely on the exit code.
- **Only `qm9`/`carbon` have baseline's root cause outstanding.** The
  finding above is disclosed, not diagnosed — nobody has traced *why*
  baseline's RNG consumption or batch composition differs on small
  structures yet.
