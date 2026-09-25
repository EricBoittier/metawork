#!/usr/bin/env bash
# Quick 1-epoch smoke test of madcore/options-pet-memmap.yaml against
# .venv-blackwell before committing to the full training run (see
# madcore-lyra-train.sbatch) -- validates the config/data/environment on a
# short debug-qos allocation instead of burning hours of normal-qos queue
# time on a config typo.
#
# Submit per-partition:
#   sbatch --partition=rtx6000 --job-name=madcore-smoke-rtx6000 pipeline-bench/slurm/madcore-lyra-smoke.sbatch
#   sbatch --partition=b200 --job-name=madcore-smoke-b200 pipeline-bench/slurm/madcore-lyra-smoke.sbatch
#SBATCH --qos=debug
#SBATCH --time=00:30:00
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

"$MTT" train /home/boittier/metawork/madcore/options-pet-memmap.yaml \
  -o smoke-model.pt \
  -r 'architecture.training.num_epochs=1'
status=$?

echo
echo "job finished at $(date) with status $status"
exit $status
