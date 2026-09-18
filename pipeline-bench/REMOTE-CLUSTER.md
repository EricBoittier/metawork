# Running the pipeline benchmark on a remote cluster

Handoff document. `README.md` covers what the harness is and how to run it
on this laptop; this file covers everything that is different when the
machine is somewhere else, and what state the work was left in.

## What you are running

One **cell** is a single `benchmarks/benchmark_pipeline.py` run: a real PET
training loop, `METATRAIN_TIMING=1`, against one git worktree of one branch
under test. A cell writes a `.json` (parsed timings, host, GPU, commit shas)
and the raw `.log`. Snakemake builds the worktrees, runs the matrix, and
aggregates the JSONs into `results/summary.md`.

The default matrix in `config.yaml` is

    6 variants x 3 datasets x 3 worker counts x 1 batch size x 2 repeats
    = 108 cells, 6 epochs each

with a per-cell timeout of 3600 s. **Read the calibration section before
submitting**: 108 cells do not necessarily fit in the 12 h the job spec asks
for.

## State as of 2026-09-18

All six variant branches are pushed to `EricBoittier/metatrain` and the
harness resolves them from `origin`:

| variant | ref | tip |
| --- | --- | --- |
| `baseline` | `origin/perf/1-pipeline-benchmark` | `f8f37091` |
| `persistent` | `origin/perf/4-persistent-workers` | `59f9a4c8` |
| `timed` | `origin/pr/collate-transform-timing` | `5166057c` |
| `pinned` | `timed` + `origin/pr/pin-batches` | `a332e897` merged in |
| `transport` | `origin/pr/system-tensor-transport` | `48363d5d` |
| `everything` | `origin/perf/5-batch-transport` | `4aade09c` |

Only a CPU smoke cell has ever been run (`results/` holds one: qm9,
workers=0, 2 epochs, ~640 atoms/s on an RTX 4060 Ti host). **No full matrix
has been run anywhere**, on GPU or otherwise. Treat the existing `results/`
as scratch and delete it before a real run, or the aggregate will mix a
2-epoch CPU cell into the tables.

The two ZBL branches are unrelated to the benchmark and not part of any
variant: `pr/zbl-training-coverage` is
[#1276](https://github.com/metatensor/metatrain/pull/1276), and
`zbl-setup-on-1267` is parked locally until #1267 merges.

The harness lives in `EricBoittier/metawork` on `main`, and a clone of that
plus the `metatrain` submodule is everything you need.

## What the cluster needs

Three things, and they are separate on purpose:

1. **A metatrain clone**, writable, with `EricBoittier/metatrain` as a remote
   named `origin` — the `metatrain` submodule of metawork already is one. The
   harness runs `git worktree add` inside it, so it needs the actual
   repository, not an export. The submodule lands detached at whatever
   commit metawork happens to record, which does not matter: the variants
   are built as worktrees of refs fetched from `origin`, and the checkout
   itself only supplies the datasets under `tests/resources`. It must be
   **installed editable**
   into the run environment (see below) because `setuptools_scm` only writes
   `src/metatrain/_version.py` into the source tree that way, and
   `metatrain/__init__.py` imports `__version__` from it. Without that file
   every cell dies at import.
2. **A run environment** with metatrain's dependencies: torch (matching the
   cluster's CUDA), metatensor, metatomic, ase, and `psutil` (optional — the
   benchmark degrades to empty memory columns without it). This is
   `config.yaml: python`. The *installed* metatrain in it is shadowed at run
   time: `run_cell.py` puts `<worktree>/src` on `PYTHONPATH`, so each cell
   imports the variant's code, not the env's. The install exists for the
   dependencies and for `_version.py`.
3. **A snakemake environment**, deliberately separate, holding only
   `snakemake>=8`, `pyyaml`, `pandas`, `matplotlib` (`requirements.txt`).
   Keeping it out of the run environment means snakemake's dependency
   resolution can never perturb what the benchmark measures.

Which environment runs what is not arbitrary, and the Snakefile gets it
right on your behalf: `make_si_bulk.py` and the benchmark itself need the
run env, while `make_worktree.py`, `run_cell.py` and `aggregate.py` are
driven by the snakemake env. Only when you call a script by hand do you have
to pick.

## Porting it, step by step

```bash
# 1. the harness and the repo under test. metatrain is a submodule of
#    metawork pointing at the fork, so init it rather than cloning it
#    separately -- but only it: there are 13 submodules and you want one.
git clone https://github.com/EricBoittier/metawork.git
cd metawork
git config submodule.metatrain.url \
  https://github.com/EricBoittier/metatrain.git   # skip if you have an SSH key
git submodule update --init metatrain
git -C metatrain fetch origin                     # brings the variant branches

# 2. run environment (adjust the torch index for the cluster's CUDA)
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -e metatrain psutil
ls metatrain/src/metatrain/_version.py    # must exist now

# 3. snakemake environment
cd pipeline-bench
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# 4. point the harness at this machine (see the table below)
$EDITOR config.yaml

# 5. prove it works, on CPU (the cell itself takes seconds; the first run
#    also has to build the worktree, which is a full checkout)
.venv/bin/snakemake --profile profiles/local --config smoke=true
cat results/summary.md
```

Both venvs were Python 3.12 locally. Anything snakemake 8 supports is fine,
but keep the run environment on a version with wheels for your torch build.

### What to edit

Every path in `config.yaml` is absolute and points at this laptop:

| file | field | local value | change to |
| --- | --- | --- | --- |
| `config.yaml` | `metatrain` | `/home/ericb/metawork/metatrain` | the clone from step 1 |
| `config.yaml` | `python` | `/home/ericb/metawork/.venv/bin/python` | the run env from step 2 |
| `config.yaml` | `datasets.qm9.path` | `.../metatrain/tests/resources/qm9_reduced_100.xyz` | same file in the new clone |
| `config.yaml` | `datasets.carbon.path` | `.../metatrain/tests/resources/carbon_reduced_100.xyz` | same file in the new clone |
| `config.yaml` | `sweep.device` | `cuda` | leave, unless benchmarking CPU |
| `config.yaml` | `resources.cpus` | `8` | match `--cpus-per-task` |

`datasets.si216` is generated into `data/si_bulk_216.xyz` by a rule, so it
needs no path. `worktrees`, `results` and `data` are relative to
`pipeline-bench/` and can stay.

Keep `sweep.workers` including `0`: the zero-worker cells are the control
that shows what the worker processes are actually buying, and they are also
the only cells that cannot be perturbed by the host's core count.

## Calibrate before you submit

The 12 h in `etc/hpc-jobs/examples/pipeline-bench.yaml` divided by 108 cells
is **6.7 minutes per cell**, while `sweep.timeout_s` lets a single cell run
for 60 minutes before it is killed. If the si216 cells run long, the job
dies partway through the matrix. Time one cell of the heaviest dataset
first, on the target GPU, and do the arithmetic:

```bash
srun --gres=gpu:1 --cpus-per-task=8 --time=00:30:00 --pty bash   # or your equivalent
cd pipeline-bench
# note the run env here: make_si_bulk needs ase, the snakemake venv has none
../.venv/bin/python scripts/make_si_bulk.py --out data/si_bulk_216.xyz \
  --n-cells 3 --n-structures 64 --seed 0
time .venv/bin/python scripts/run_cell.py \
  --worktree worktrees/everything \
  --python ../.venv/bin/python \
  --dataset data/si_bulk_216.xyz --key energy \
  --num-workers 4 --batch-size 8 --device cuda --epochs 6 \
  --out /tmp/cell.json --log /tmp/cell.log
```

(That needs `worktrees/everything` to exist; either run the smoke first or
call `.venv/bin/python scripts/make_worktree.py --repo ../metatrain --path
worktrees/everything --refs origin/perf/5-batch-transport`.)

Then either raise `time:` in the job spec, or cut the matrix. Cheapest cuts,
in the order I would make them: `repeats: 2 -> 1` halves everything;
dropping `transport` and `pinned` leaves the four variants that correspond to
opened PRs; dropping `carbon` keeps one light and one heavy dataset.

Nothing is lost by running out of time, though — see resuming, below.

## Submitting

On kuma, the job spec and wrapper already exist:

```bash
./submit.sh              # prepares runs/<timestamp>_pipeline-bench/job.sbatch
./submit.sh --submit     # ...and calls sbatch
```

`etc/hpc_run.py` only knows kuma's QOS limits (it skips validation for
unknown clusters) and it assumes the metawork layout: it activates
`<repo root>/.venv` and `cd`s to `<repo root>/pipeline-bench`. It also has no
`module load` support — if the cluster needs modules, put them at the top of
the `command:` block in the YAML.

On any other cluster, skip it and write the sbatch directly; this is all it
generates:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=pipeline-bench
#SBATCH --output=slurm-%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail
# module load cuda/12.4    # if your cluster needs it
source /path/to/metawork/.venv/bin/activate
cd /path/to/metawork/pipeline-bench
.venv/bin/snakemake --profile profiles/local --cores 1 --resources gpu=1
```

`--cores 1 --resources gpu=1` is the important part and is not a performance
oversight: cells are serialized so that each one has the card to itself. Two
cells sharing a GPU produce numbers that cannot be compared with each other
or with anything else.

`profiles/slurm/` submits one sbatch per cell instead (`--profile
profiles/slurm`). Only use it if the partition gives whole GPUs per job, and
fix `slurm_partition`, `qos` and `--gres` in it first — they are guesses.

## Resuming, and collecting results

Snakemake keys off the output files, so a job that hits its walltime loses
only the cell it was in. Resubmit the same command and it runs the missing
cells. `rerun-incomplete` is already on in both profiles. To force a rerun,
delete the relevant `results/cells/<variant>/...` JSONs (or all of
`results/`).

What to bring home:

- `results/summary.md` — median over repeats, speedup against `baseline`
- `results/cells.csv` — one row per cell, including GPU name, torch version,
  commit shas, wall time, memory
- `results/stages.csv` — the per-stage timings (loader, unpack, h2d,
  serialize, and the per-transform breakdown from the `timed` variants)
- `results/cells/**/*.log` — keep these; they are the only record if a cell
  behaves oddly

Optional plot, needs matplotlib:

```bash
.venv/bin/python scripts/plot.py --cells-csv results/cells.csv \
  --out results/atoms_per_s.png
```

The whole `results/` tree is a few MB at most; `rsync` it back and re-run
`scripts/aggregate.py` locally if you want to re-cut the tables.

## Gotchas worth knowing before they bite

- **`_version.py`** — the single most likely failure. `make_worktree.py`
  copies it from the live checkout into each worktree, so if the clone was
  not installed editable, every cell fails at import. Check for the file.
- **The `pinned` variant merges two branches.** `pr/pin-batches` and
  `pr/collate-transform-timing` both add kwargs at the same `DataLoader`
  sites, so the merge conflicts by construction; `make_worktree.py` resolves
  add/add conflicts by keeping both sides. If a future rebase changes those
  branches, check `worktrees/pinned` actually has both changes before
  trusting its numbers.
- **Worktrees are cached by sha** (`.variant-shas`). If you force-push a
  branch, the worktree is rebuilt on the next run — but only if the harness
  can see the new sha, so `git -C metatrain fetch origin` first.
- **Datasets come from the live checkout, not the worktrees**, which is what
  makes the comparison fair. Don't repoint them at per-variant copies.
- **`nvidia-smi` is deliberately not used.** GPU-utilization sampling was
  removed because it cost more than it measured; memory comes from `psutil`
  RSS around the run. If you want GPU-side memory, add
  `torch.cuda.max_memory_allocated()` to the benchmark rather than polling.
- **Host core count used to leak into PET's RNG stream**, through the worker
  seed each `DataLoader` drew from the global generator. Fixed on
  `perf/4-persistent-workers` and everything stacked on it, which is every
  variant except `baseline`. It moves trained values, not throughput, so
  it does not invalidate timings — but it does mean losses from different
  variants are not comparable run to run.

## What this was meant to answer

The open question the matrix exists to settle: after persistent workers hid
most of collation, `unpack` was the largest remaining pipeline cost (8-10 ms
per batch on heavy periodic systems), and the per-transform timing in the
`timed` variant should identify the largest *deterministic* repeated
transform — the one worth hoisting out of per-epoch collation into a
preparation step owned by `Trainer.setup()` once
[#1267](https://github.com/metatensor/metatrain/pull/1267) lands. `transport`
and `pinned` are the speculative answers; `everything` is the integration
branch, [#1275](https://github.com/metatensor/metatrain/pull/1275).

Write that conclusion up from `results/stages.csv` — it is the input to the
next PR, and the reason the matrix is worth the GPU hours.
