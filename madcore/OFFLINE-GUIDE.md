# Submitting MAD-CORE benchmark jobs without an agent on the login node

Everything below is plain `ssh` + `sbatch` + `squeue` -- no Claude Code, no
IDE, nothing running interactively on a login node. Read `README.md` and
`HANDOFF-benchmark.md` first for what MAD-CORE/the converter/the split
actually are; this is just "which commands do I type, on which cluster."

## Two clusters, two environments

| | kuma | lyra |
| --- | --- | --- |
| GPUs | L40S (sm_89), H100 (sm_90) | b200 (sm_100), rtx6000 Blackwell (sm_120) |
| Python env | shared `/home/boittier/metawork/.venv` (torch 2.5.1+cu121) | separate `/home/boittier/metawork/.venv-blackwell` (torch 2.14+cu130) |
| Needs building? | No, already there | Yes, once (below) -- cu121 cannot run on Blackwell at all (`no kernel image is available for execution on the device`) |

**Do not point kuma jobs at `.venv-blackwell` or vice versa.** The two are
deliberately separate so kuma's environment is never touched by lyra-specific
changes.

`ssh kuma` / `ssh lyra` (or whatever your `~/.ssh/config` aliases are) to get
a login shell on either, then `cd ~/metawork` and use the commands below.

## One-time: build the Blackwell venv (lyra only)

```bash
cd ~/metawork/pipeline-bench
sbatch --partition=b200 --job-name=build-blackwell-venv-b200 \
  slurm/build-blackwell-venv.sbatch
```

Takes ~10-35 min once queued (mostly torch download + a from-source
metatensor-torch/metatomic-torch build). `rtx6000` works too
(`--partition=rtx6000`), but `b200`'s queue has been consistently shorter.
Check `smoke-slurm/build-blackwell-venv-<partition>-<jobid>.out` for
progress; it ends with a real GPU matmul as a sanity check. If it fails,
see **Troubleshooting** below -- every failure mode we actually hit is
documented inline in the script itself at the point it's handled.

You only need to do this once; `.venv-blackwell` persists on the shared
filesystem across sessions. Re-run it (after `rm -rf ~/metawork/.venv-blackwell`
first) only if it stops working, e.g. after a driver/module update.

## Data (once, either cluster -- shared filesystem)

```bash
MC_TOKEN=<token from Eric> bash ~/metawork/etc/download-madcore-extxyz.sh --keep-gz   # ~11 GB
```

Already done as of this writing (`~/data/madcore/*.extxyz.gz`, SHA-256
verified against the record's `index.json`). Skip this if those files are
already there.

## Build the converter (once, per cluster -- it's a compiled binary)

```bash
cd ~/metawork/metatomic-core-examples
bash build.sh
```

Needs `cargo`, `zlib` headers, a C/C++ compiler; does **not** need HDF5
headers for the two binaries this workflow uses (`build.sh` skips the one
HDF5-dependent C++ example gracefully if `hdf5-devel` isn't installed, which
it currently isn't on lyra -- runtime `.so` present, dev headers not, no
root to fix it). Produces `build/madcore-to-diskdataset` and
`build/madcore-scan`.

## Convert a coreset level to MemmapDataset (once per level you want)

```bash
cd ~/metawork/metatomic-core-examples
./build/madcore-to-diskdataset --split ../madcore/splits/madcore-split-seed0.npz \
    -o ~/data/madcore/memmap-2pow18 --max-rows 262144 ~/data/madcore/madcore_*.extxyz.gz
```

`--max-rows 262144` is the level-18 coreset (start here per the handoff).
Drop `--max-rows` for the full 2^24. Takes well under a minute for level-18;
no GPU needed, safe to run directly on a login node (it's a lightweight
Rust binary, not a training job).

## Submitting benchmark/training jobs

**kuma** -- pipeline-bench timing comparison (baseline vs `everything`,
real MAD-CORE data, existing shared env):

```bash
cd ~/metawork/pipeline-bench
sbatch --partition=l40s --job-name=madcore-kuma-l40s slurm/madcore-kuma-smoke.sbatch
```

Results: `smoke-slurm/madcore-kuma-l40s-<jobid>.out` (stage-by-stage timing
table for each variant) plus per-variant JSON/log in `smoke-slurm/`.

**lyra** -- quick 1-epoch sanity check before committing to a full run:

```bash
cd ~/metawork/pipeline-bench
sbatch --partition=b200 --job-name=madcore-smoke-b200 slurm/madcore-lyra-smoke.sbatch
```

**lyra** -- the actual training benchmark (per `madcore/README.md`'s
"Suggested benchmarks"):

```bash
cd ~/metawork/pipeline-bench
sbatch --partition=b200 --job-name=madcore-train-b200 slurm/madcore-lyra-train.sbatch
```

Both lyra scripts run `mtt train madcore/options-pet-memmap.yaml` from inside
`~/data/madcore/memmap-2pow18` using `.venv-blackwell`'s `mtt`. Edit the
`.sbatch` file directly to point at a different memmap directory (e.g. a
different coreset level) or to override the options file.

## Checking on things

```bash
squeue -u $USER                          # what's queued/running
squeue --start -j <jobid>                # Slurm's estimated start time (backfill scheduler)
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed
tail -f pipeline-bench/smoke-slurm/<name>-<jobid>.out
```

Both clusters' queues can be busy with other groups' jobs (kuma: `forina`;
lyra: `chorna`/`corradin`/`abbott`/`pegolo`, some of whom look like the actual
MAD-CORE/PET-MAD authors). If `squeue --start` shows a start time many hours
out, try the other partition on that cluster (`b200` vs `rtx6000`, `l40s` vs
`h100`) -- load shifts unpredictably.

## Troubleshooting

**`CUDA error: no kernel image is available for execution on the device`**
You used the shared `.venv` (cu121) on a Blackwell GPU (lyra). Use
`.venv-blackwell` there instead -- see `slurm/madcore-lyra-*.sbatch` for how
they invoke `mtt`/`run_cell.py` with the right python.

**`CUDA out of memory` on rtx6000 that doesn't reproduce on b200 with the
same config:** check `nvidia-smi` inside the job (both smoke scripts already
print `nvidia-smi -L`) -- an OOM at the very first training step with
tens of GB already "in use" before your own model/batch could plausibly
need that much usually means the GPU is shared with another job, not that
your config is actually too big. Retry, possibly on a different node/partition.

**`error: No solution found ... metatensor-torch depends on torch==2.5.*`**
while rebuilding `.venv-blackwell`: this means a previous build's cached
wheel metadata (keyed by source content, not by which torch was installed
at build time) got reused. `rm -rf ~/metawork/.venv-blackwell` and resubmit
`build-blackwell-venv.sbatch` -- it already includes `--reinstall-package`/
`--refresh-package` to prevent this on a clean venv, but the fix doesn't
retroactively repair an already-poisoned one.

**`benchmark_pipeline.py: error: unrecognized arguments: --seed`** -- fixed
as of this branch (`run_cell.py` now probes `--help` before forwarding
`--seed`); if you see this again, the worktree in question
(`pipeline-bench/worktrees/<variant>/`) predates that flag in its own
`benchmark_pipeline.py`, which is expected and harmless (`run_cell.py` just
won't pass `--seed` to it).

**Build script errors mentioning `nvhpc`, `CUDA_HOME`, or `nvc++`** on lyra:
these are all handled already inside `build-blackwell-venv.sbatch` (each
fix has a comment explaining the underlying issue: NVHPC's directory layout
splits the compiler from the CUDA headers/libs, and loading that module
puts `nvc++` ahead of `gcc`/`g++` on PATH unless overridden). If the script
itself needs changing, that's on branch `infra/blackwell-torch-workaround`.

## Branches

- `bench/medium-hardware` (this one): the MAD-CORE split/converter/options
  file, plus everything in this guide.
- `infra/blackwell-torch-workaround`: just the `.venv-blackwell` build
  script, if you want it in isolation (e.g. to bring into another branch).

Both live on `origin` = `EricBoittier/metawork`. Never push to the
`metatensor/*` upstream remotes of the vendored ecosystem repos, and don't
commit inside the `metatomic/` checkout (detached HEAD with local edits --
build from it, don't commit to it).
