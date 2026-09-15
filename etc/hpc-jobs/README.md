# HPC job runner

`etc/hpc_run.py` wraps a shell command (a metatrain training run, an i-pi/
LAMMPS MD run, anything) with a SLURM batch script and a reproducibility
manifest, then leaves it to you to submit -- no job DAG, no database.

## Usage

```
.venv/bin/python etc/hpc_run.py etc/hpc-jobs/examples/train-soap-bpnn.yaml
```

This creates `runs/<timestamp>_<name>/` containing:

- `manifest.yaml` -- the resolved job config, the exact command, the
  branch/commit/dirty-state of every repo in this workspace, and the
  python/torch/CUDA versions in the venv used, all captured at prepare time.
- `job.sbatch` -- the generated SLURM batch script.
- copies of any `inputs:` files listed in the spec (e.g. the training
  options file), so editing the original later doesn't change what this run
  actually used.

It does **not** submit anything by itself. Either run the printed `sbatch`
command yourself, or pass `--submit` to have the script call `sbatch`
directly (only works on a node that actually has SLURM, e.g. a real Kuma
login/compute node, not a workstation) -- the resulting job ID is then
written back into `manifest.yaml`.

## Writing a job spec

See `etc/hpc_run.py`'s module docstring for the full field list, and
`examples/` for a training and an MD job. The important ones:

```yaml
name: my-run
type: train            # free-form label, just goes in the manifest
cluster: kuma
qos: debug              # validated against etc/hpc_run.py's CLUSTERS table
time: "01:00:00"
nodes: 1
gpus: 1
cpus_per_task: 8
venv: shared             # shared (.venv) or upet (.venv-upet)
workdir: etc/lorem-parity  # cwd for `command`, relative to the repo root
inputs: [options.yaml]     # snapshotted into the run directory
command: |
  mtt train options.yaml -o model.pt
```

If `time`/`nodes`/`gpus`/`cpus_per_task` exceed the chosen QOS's limits, the
script refuses to prepare the run rather than let a submission get rejected
(or silently killed) by SLURM later.

## Kuma QOS reference

| QOS      | priority | max wall-time | max resources          |
|----------|----------|----------------|-------------------------|
| `normal` | normal   | 3-00:00:00     | 8 nodes                 |
| `long`   | low      | 7-00:00:00     | 8 nodes                 |
| `build`  | high     | 04:00:00       | 1 node, 0 GPU, 16 cores |
| `debug`  | high     | 01:00:00       | 2 GPU                   |

Use `build` for compile-only work that doesn't need a GPU, `debug` for a
quick GPU smoke test, and `normal`/`long` for real training or production MD
runs.

## Adding another cluster

Add an entry to `CLUSTERS` in `etc/hpc_run.py` with its QOS names and
limits, then set `cluster: <name>` in a job spec. Unknown clusters just skip
validation rather than erroring, so a spec still works before you've filled
in its table.
