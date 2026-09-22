#!/usr/bin/env bash
# Runs the `distributed` Snakemake target (config.yaml's `distributed:`
# block: baseline vs everything x world_size 1/2/4/8 x repeats, on the
# omol_billion_proxy dataset) for the NVIDIA compute-grant proposal
# ("Scaling interatomic potential with universal billion-structure
# dataset", target resource 8xH100).
#
# kuma's H100 nodes are 4 GPUs each -- world_size=8 needs 2 nodes. This
# allocates 2 nodes x 4 GPUs/node up front; smaller world_size cells
# (1/2/4) fit inside a single node of that same allocation (srun packs
# onto as few nodes as it can), so only the world_size=8 cells actually
# exercise the cross-node InfiniBand path -- see the note in config.yaml.
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks=8
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

.venv/bin/snakemake distributed --profile profiles/local --cores 1 \
  --resources gpu=8 --rerun-incomplete
status=$?

echo
cat results/distributed_summary.md 2>/dev/null

echo
echo "job finished at $(date) with status $status"
exit $status
