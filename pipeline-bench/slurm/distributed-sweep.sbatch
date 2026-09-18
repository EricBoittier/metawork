#!/usr/bin/env bash
# Runs the formalized `distributed` Snakemake target (variant x world_size x
# repeat sweep from config.yaml's `distributed:` block), not the ad hoc
# multigpu-smoke.sbatch script. Needs an allocation with as many GPUs+tasks
# as the largest world_size in that sweep, since `run_cell.py --world-size N`
# calls `srun --ntasks N` for the distributed cells, using this allocation.
#SBATCH --qos=debug
#SBATCH --time=00:30:00
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

echo "== job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), ntasks=$SLURM_NTASKS =="
date
nvidia-smi -L || echo "nvidia-smi not available"

.venv/bin/snakemake distributed --profile profiles/local --cores 1 \
  --resources gpu=2 --rerun-incomplete
status=$?

echo
cat results/distributed_summary.md 2>/dev/null

echo
echo "job finished at $(date) with status $status"
exit $status
