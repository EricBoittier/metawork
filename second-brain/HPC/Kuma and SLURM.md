---
tags: [hpc, slurm, kuma]
---
# Kuma and SLURM

```bash
kuma            # alias: ssh boittier@kuma.hpc.epfl.ch
lyra            # alias: ssh boittier@lyra.hpc.epfl.ch
```

Full cluster details (partitions, Lyra, pricing): [[EPFL SCITAS (Kuma, Lyra, Jed)]] · generic commands: [[SLURM cheatsheet]]

**No default partition** — every job needs `-p h100 | l40s | mig12gb | mig24gb`. RAM is fixed at 5900 MB/core.

## QOS
| QOS | Priority | Max time | Limits | Use for |
|---|---|---|---|---|
| `debug` | high | 1 h | ≤ 2 GPUs | smoke tests, compiling CUDA kernels (`nvcc`) |
| `build` | high | 4 h | 1 node, 0 GPU, 16 cores | CPU-only compiles |
| `normal` | normal | 3 days | ≤ 8 nodes | production |
| `long` | low | 7 days | ≤ 8 nodes | long jobs that can wait |

Login node has no `nvidia-smi`/`nvcc` → building `*-torch` CUDA kernels needs a `debug` job.

## SLURM commands
```bash
squeue -u $USER                         # my jobs
squeue -u $USER --start                 # estimated start times
sbatch job.sbatch
scancel <jobid>                         # scancel -u $USER = all mine
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
scontrol show job <jobid>
sinfo -s                                # partitions overview
# interactive GPU shell:
srun -p h100 --qos=debug --gres=gpu:1 --cpus-per-task=8 --time=01:00:00 --pty bash
```

## metawork job runner
Wraps a command in an sbatch script + reproducibility manifest (repo commits, torch/CUDA versions).
```bash
cd ~/Documents/metawork
.venv/bin/python etc/hpc_run.py etc/hpc-jobs/examples/train-soap-bpnn.yaml            # prepare only
.venv/bin/python etc/hpc_run.py etc/hpc-jobs/examples/train-soap-bpnn.yaml --submit   # prepare + sbatch
```
Creates `runs/<timestamp>_<name>/` with `manifest.yaml`, `job.sbatch`, snapshotted inputs.

Spec skeleton:
```yaml
name: my-run
type: train
cluster: kuma
partition: h100        # required on kuma
qos: debug
time: "01:00:00"
nodes: 1
gpus: 1
cpus_per_task: 8
venv: shared            # shared (.venv) or upet (.venv-upet)
workdir: etc/lorem-parity
inputs: [options.yaml]
command: |
  mtt train options.yaml -o model.pt
```
Pipeline benchmark: `pipeline-bench/submit.sh [--submit]` (see `pipeline-bench/REMOTE-CLUSTER.md`).

Related: [[CSCS Alps]], [[SSH and tunnels]]
