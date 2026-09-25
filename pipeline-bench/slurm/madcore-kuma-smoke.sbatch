#!/usr/bin/env bash
# Real-data (MAD-CORE) pipeline-bench comparison on kuma, using the shared
# /.venv/ (torch 2.5.1+cu121) directly -- kuma's L40S (sm_89) / H100 (sm_90)
# are within that build's supported arch range, unlike lyra's Blackwell
# GPUs, so no separate venv is needed here.
#
# Submit per-partition, e.g.:
#   sbatch --partition=l40s --job-name=madcore-kuma-l40s slurm/madcore-kuma-smoke.sbatch
#   sbatch --partition=h100 --job-name=madcore-kuma-h100 slurm/madcore-kuma-smoke.sbatch
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --chdir=/home/boittier/metawork/pipeline-bench
#SBATCH --output=/home/boittier/metawork/pipeline-bench/smoke-slurm/%x-%j.out
#SBATCH --error=/home/boittier/metawork/pipeline-bench/smoke-slurm/%x-%j.err

set -uo pipefail

echo "== job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), partition $SLURM_JOB_PARTITION =="
date
nvidia-smi -L || echo "nvidia-smi not available"

DATASET=data/madcore_scaling.xyz
KEY=energy
overall=0

for variant in baseline everything; do
  out_json="smoke-slurm/madcore-${variant}-${SLURM_JOB_ID}.json"
  out_log="smoke-slurm/madcore-${variant}-${SLURM_JOB_ID}.log"
  echo
  echo "== variant: $variant =="
  .venv/bin/python scripts/run_cell.py \
    --worktree "worktrees/${variant}" \
    --python /home/boittier/metawork/.venv/bin/python \
    --dataset "$DATASET" --key "$KEY" \
    --num-workers 0 --batch-size 8 --device cuda --epochs 3 \
    --timeout-s 1200 \
    --variant "$variant" --dataset-name madcore --repeat 0 \
    --out "$out_json" --log "$out_log"
  status=$?
  echo "-- $variant exit=$status --"
  overall=$((overall + status))
  echo "-- log tail --"
  tail -n 25 "$out_log" 2>/dev/null
done

echo
echo "job finished at $(date) with status $overall"
exit $overall
