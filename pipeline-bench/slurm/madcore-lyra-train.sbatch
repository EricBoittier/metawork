#!/usr/bin/env bash
# Trains PET on the MAD-CORE level-18 coreset (madcore/options-pet-memmap.yaml)
# using .venv-blackwell (see slurm/build-blackwell-venv.sbatch) -- the shared
# /.venv/ can't run on this cluster's Blackwell GPUs at all (torch==2.5.1+cu121
# tops out at sm_90). Data: ~/data/madcore/memmap-2pow18, built via
# metatomic-core-examples/build/madcore-to-diskdataset per madcore/README.md.
#
# Submit per-partition:
#   sbatch --partition=rtx6000 --job-name=madcore-train-rtx6000 pipeline-bench/slurm/madcore-lyra-train.sbatch
#   sbatch --partition=b200 --job-name=madcore-train-b200 pipeline-bench/slurm/madcore-lyra-train.sbatch
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --chdir=/home/boittier/data/madcore/memmap-2pow18
#SBATCH --output=/home/boittier/metawork/pipeline-bench/smoke-slurm/%x-%j.out
#SBATCH --error=/home/boittier/metawork/pipeline-bench/smoke-slurm/%x-%j.err

set -uo pipefail

echo "== job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), partition $SLURM_JOB_PARTITION =="
date
nvidia-smi -L || echo "nvidia-smi not available"

MTT=/home/boittier/metawork/.venv-blackwell/bin/mtt

"$MTT" train /home/boittier/metawork/madcore/options-pet-memmap.yaml
status=$?

echo
echo "job finished at $(date) with status $status"
exit $status
