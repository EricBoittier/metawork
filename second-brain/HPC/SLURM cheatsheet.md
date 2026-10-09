---
tags: [hpc, slurm, cheatsheet]
---
# SLURM cheatsheet

Cluster specifics (partitions, QOS, accounts): [[EPFL SCITAS (Kuma, Lyra, Jed)]] · [[CSCS Alps]] · [[sciCORE]]

## Submit / watch / cancel
```bash
sbatch job.sh                      # → Submitted batch job 123
sbatch --test-only job.sh          # validate + estimated start, no submit
squeue --me                        # my jobs (or -u $USER)
squeue --me --start                # estimated start times
squeue --me -o "%.10i %.20j %.8T %.10M %.6D %R"   # compact view with reason
scancel 123 | scancel --me | scancel -n jobname
scontrol show job 123              # full details, why pending
scontrol hold 123 / release 123
scontrol update job=123 TimeLimit=2:00:00   # can only lower usually
```

## After the fact
```bash
sacct -j 123 --format=JobID,JobName,State,Elapsed,MaxRSS,ReqMem,AllocTRES%40,ExitCode
sacct --me -S 2026-10-01 -X        # all my jobs since date, one line each
seff 123                           # CPU/mem efficiency summary (if installed)
```

## Interactive
```bash
srun -p <part> -q debug --gres=gpu:1 -c 8 -t 01:00:00 --pty bash
salloc -p <part> --gres=gpu:1 -t 1:00:00     # allocation, then `srun ...` inside it
ssh <node>                         # allowed while you have a job on it → nvidia-smi, htop
```

## Resource flags
| Flag | Meaning |
|---|---|
| `-N/--nodes` | nodes |
| `-n/--ntasks`, `--ntasks-per-node` | MPI ranks / processes (1 per GPU for DDP) |
| `-c/--cpus-per-task` | threads per rank (set `OMP_NUM_THREADS` to match) |
| `--gres=gpu:N` / `--gpus-per-task=1` / `--gpus-per-node=N` | GPUs |
| `--mem`, `--mem-per-cpu` | RAM (SCITAS fixes it per core) |
| `-t D-HH:MM:SS` | wall time |
| `-p`, `-q`, `-A` | partition, QOS, account |
| `--exclusive` | whole node |
| `-o/-e out-%x-%j.log` | `%x` name, `%j` job id, `%A_%a` array |
| `--signal=B:USR1@300` | send USR1 5 min before time limit (checkpoint!) |

## Useful env vars inside a job
`SLURM_JOB_ID`, `SLURM_SUBMIT_DIR`, `SLURM_JOB_NODELIST`, `SLURM_NNODES`, `SLURM_NTASKS`, `SLURM_PROCID` (global rank), `SLURM_LOCALID` (local rank), `SLURM_CPUS_PER_TASK`, `SLURM_ARRAY_TASK_ID`, `CUDA_VISIBLE_DEVICES`.
First node: `scontrol show hostnames $SLURM_JOB_NODELIST | head -n1`.

## Job arrays (parameter sweeps)
```bash
#SBATCH --array=0-19%4             # 20 tasks, max 4 at once
cfg=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" configs.txt)
python run.py --config $cfg
```

## Dependencies (pipelines)
```bash
a=$(sbatch --parsable prep.sh)
b=$(sbatch --parsable --dependency=afterok:$a train.sh)
sbatch --dependency=afterany:$b analyse.sh
```

## Checkpoint & requeue on time limit
```bash
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
trap 'echo "time almost up"; kill -USR1 $PID; wait $PID; scontrol requeue $SLURM_JOB_ID' USR1
python train.py --resume & PID=$!; wait $PID
```

## Pending reasons
| Reason | Meaning |
|---|---|
| `Priority` | others ahead of you; wait |
| `Resources` | you're next, waiting for nodes |
| `QOSMaxWallDurationPerJobLimit` | `-t` over QOS limit |
| `AssocGrpGRES` / `QOSMaxGRESPerUser` | GPU cap for your account/QOS |
| `ReqNodeNotAvail` | maintenance reservation — shorten `-t` to fit before it |

Related: [[Distributed overview]], [[Clusters overview]]
