#!/usr/bin/env bash
# Real multi-GPU DDP smoke test, on the size-diverse omol_proxy dataset with
# atoms-per-batch batching. Per metatrain's own multi-GPU example
# (metatrain/examples/1-advanced/05-multi-gpu.py, whose sample log shows it
# was run on this exact cluster): distributed training auto-enables whenever
# SLURM_NTASKS > 1 (resolve_distributed() in
# metatrain/utils/distributed/slurm.py), and Trainer.train() does its own
# DistributedDataParallel wrapping -- benchmark_pipeline.py itself needs no
# code changes for DDP, just launching via `srun --ntasks=N` with one task
# per GPU. `--gres=gpu:N` and `--nodes=1 --gpus=N` are both rejected by this
# cluster's job_submit plugin (see gpu-smoke.sbatch); `--gpus-per-node` does
# not have that problem.
#
# Submit with the desired world size, e.g.:
#   sbatch --partition=l40s --ntasks=2 --gpus-per-node=2 slurm/multigpu-smoke.sbatch
#SBATCH --qos=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --chdir=/home/boittier/metawork/pipeline-bench
#SBATCH --output=/home/boittier/metawork/pipeline-bench/smoke-slurm/%x-%j.out
#SBATCH --error=/home/boittier/metawork/pipeline-bench/smoke-slurm/%x-%j.err

set -uo pipefail

echo "== job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), partition $SLURM_JOB_PARTITION, ntasks=$SLURM_NTASKS =="
date
nvidia-smi -L || echo "nvidia-smi not available"

export PYTHONPATH="worktrees/everything/src"
export METATRAIN_TIMING=1

srun --output="smoke-slurm/%x-%j.rank%t.out" \
  /home/boittier/metawork/.venv/bin/python worktrees/everything/benchmarks/benchmark_pipeline.py \
  --dataset /home/boittier/metawork/pipeline-bench/data/omol_proxy.xyz \
  --key energy --num-workers 0 --batch-size 8 --device cuda --epochs 2 \
  --max-atoms-per-batch 512
status=$?

echo
echo "== rank 0 report =="
tail -n 40 "smoke-slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.rank0.out" 2>/dev/null

echo
echo "job finished at $(date) with status $status"
exit $status
