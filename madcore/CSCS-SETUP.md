# Reproducing the madcore/LOREM setup on CSCS (clariden / daint)

State as of 2026-10-02, taken from kuma. CSCS facts are perishable; check them
against docs.cscs.ch or on the cluster.

## 1. Clone

```bash
git clone -b bench/medium-hardware https://github.com/EricBoittier/metawork
cd metawork
# metatomic and metatrain are https in .gitmodules; anything else on git@
# (sirmarcel/iris-infra, private) needs a GitHub key or a credential helper
git config --global url."https://github.com/".insteadOf git@github.com:
METAWORK_PINNED=1 bash etc/setup-metawork.sh
```

`METAWORK_PINNED=1` keeps every submodule on the commit recorded here instead
of switching to the `REPO_BRANCH` branches and pulling them; without it,
metatensor and metatomic land on `metatomic-core`, which is not what kuma
runs. The script no longer prompts for credentials, so a fork that does not
exist (e.g. `gpu-lite`) falls back to upstream instead of stalling.

If a run is interrupted mid-clone, remove the half-cloned submodule before
re-running (`rm -rf <name> .git/modules/<name>`), otherwise checkout refuses
to overwrite its files.

The submodules are recorded at what kuma's venv runs (editable installs):
metatensor `a9fc36e7` (main), metatomic `e12f2628`
(`fix/extensions-content-addressed`), metatrain `917b8031`
(`experimental/lorem`), openmm-ml `fb543e4` (main). kuma's compiled
`metatomic-torch` reports `0.2.0.dev67+dirty`, which does not map to a commit:
build it from the recorded checkout.

## 2. LOREM code (not on GitHub)

The `long_range` switch must never go to a remote, so the local branches travel
as a git bundle (30 KB, made on kuma, `~/bundles/`):

```bash
scp kuma:bundles/metatrain-lorem-local-2026-10-02.bundle ~/   # from a machine that reaches both
cd metawork/metatrain
git fetch origin                     # the bundle needs 42e25756 and 482ff780 from the fork
git fetch ~/metatrain-lorem-local-2026-10-02.bundle \
    'refs/heads/local/*:refs/heads/local/*'
for b in lorem-pipeline lorem-perf lorem-long-range-switch; do
    git config branch.local/$b.pushremote no-push-local-only   # same push block as kuma
done
git worktree add --detach ../metatrain-runs-e100 a73363f3     # what the 100-epoch runs use
```

| branch | head | content |
|---|---|---|
| `local/lorem-pipeline` | `a73363f3` | PR #1265 + switch + #1275 + no per-step sync + index_select + unpack to device (fastest; runs-e100) |
| `local/lorem-perf` | `a5f412dc` | PR #1265 + switch + batched Ewald |
| `local/lorem-long-range-switch` | `47c25cc3` | PR #1265 head 42e25756 + switch + Bernstein cutoff fix |

Without the switch, `origin/experimental/lorem-clean` (PR #1265, `324e5c56`)
has every LOREM speedup but is always long-range.

`_version.py` is generated and untracked: copy it into each worktree's
`src/metatrain/` (or `pip install -e` once) and run from the worktree with
`PYTHONPATH=<worktree>/src`, as the sbatch scripts do.

## 3. Environment

kuma's `~/metawork/.venv` (x86_64, uv, torch 2.5.1+cu121) cannot be copied:
GH200 nodes are aarch64. Rebuild in the Container Engine (image `.sqsh` on
scratch, EDF in `~/.edf/`, thin venv on `$HOME`). Versions kuma runs:

| package | kuma |
|---|---|
| torch | 2.5.1 (+cu121; take the aarch64 CUDA build of the image) |
| torch-pme | 0.5.0 |
| sphericart-torch | 1.0.9 |
| vesin | 0.6.1 |
| metatensor-torch / -core / -learn | 0.10.4 / 0.2.4 / 0.6.1 |
| metatensor-operations | editable, metatensor `a9fc36e` |
| metatomic-torch | built from metatomic `e12f2628` |
| numpy | 2.5.3 |

The `LD_LIBRARY_PATH` cudart workaround in the kuma sbatch files
(`~/data/madcore/lorem-runs/cudart`) is specific to kuma's CUDA 12.6 driver
and the cu121 wheels; drop it in the container.

## 4. Data

From kuma `/work/cosmo/boittier/kuma/madcore-lorem/`: `data/` (2.0 GB:
mixed 975 MB, periodic 922 MB, nonperiodic 53 MB, mixed-8192 51 MB) and
`cache/` (98 KB, composition + scaler per subset). On clariden, `$SCRATCH`
(iopsstor) is reaped after 14 days: keep a copy on a store/project area and
stage it.

## 5. Job scripts

`train-lorem-pbc.sbatch`, `bench-pipeline.sbatch` and friends hardcode kuma:
`--account=cosmo`, `--partition=h100`, `--qos`, `ROOT=/work/...`, the venv
path and the cudart line. On Alps also:
- `--account` is mandatory; nodes are allocated whole (4 GH200, a 1-GPU job
  bills 4): pack four runs per node with `CUDA_VISIBLE_DEVICES` (fine for
  training, not for timing benchmarks, ~6 % cross-talk);
- `--signal=USR1@300` (no `B:`), `--environment=<EDF name>`;
- daint `debug` is 30 min and one job at a time; `normal` 12 h (clariden) /
  24 h (daint).

Run history and timings: `RUNS.md`; narrative: `~/scitas/journal/2026-10-02.md`
on kuma.
