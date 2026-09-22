#!/usr/bin/env bash
# Runs the `distributed_best` Snakemake target (config.yaml's
# `distributed_best:` block): baseline vs everything, each using its own
# best-supported batching strategy (everything: max_atoms_per_batch=8192;
# baseline: fixed batch_size, since it predates that flag), on the scaled-up
# omol_billion_proxy_v2 dataset (20000 structures).
#
# Single node, world_sizes [1,2,4]: world_size=8 (cross-node) is covered by
# the fair fixed-batch sweep (distributed-sweep-h100.sbatch) already, and a
# 2-node request was queuing hours out on a fragmented-busy cluster.
#SBATCH --qos=normal
#SBATCH --time=01:30:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --chdir=/home/boittier/metawork/pipeline-bench
#SBATCH --output=/home/boittier/metawork/pipeline-bench/smoke-slurm/%x-%j.out
#SBATCH --error=/home/boittier/metawork/pipeline-bench/smoke-slurm/%x-%j.err

set -uo pipefail

echo "== job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $SLURM_NODELIST, ntasks=$SLURM_NTASKS =="
date
nvidia-smi -L || echo "nvidia-smi not available"

.venv/bin/snakemake distributed_best --profile profiles/local --cores 1 \
  --resources gpu=4 --rerun-incomplete
status=$?

echo
cat results/distributed_best_summary.md 2>/dev/null

echo
echo "job finished at $(date) with status $status"
exit $status
