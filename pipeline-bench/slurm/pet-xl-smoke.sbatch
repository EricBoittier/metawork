#!/usr/bin/env bash
# GPU smoke test for a much larger PET model ("XL": d_pet/d_head 128->512,
# d_node 256->1024, d_feedforward 256->2048, num_heads 8->16,
# num_attention_layers 2->4, num_gnn_layers 2->3 -- ~147.5M params vs the
# ~2.9M-param default, measured by instantiating both). No named XL preset
# exists in metatrain; this is a deliberately large scale-up to see where
# memory/throughput actually break down. Bypasses Snakemake (this isn't part
# of the normal sweep matrix) and calls run_cell.py directly.
#
# Submit per-partition, e.g.:
#   sbatch --partition=l40s --job-name=pet-xl-smoke-l40s slurm/pet-xl-smoke.sbatch
#   sbatch --partition=h100 --job-name=pet-xl-smoke-h100 slurm/pet-xl-smoke.sbatch
#SBATCH --qos=debug
#SBATCH --time=00:35:00
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

XL_HYPERS='{"d_pet":512,"d_head":512,"d_node":1024,"d_feedforward":2048,"num_heads":16,"num_attention_layers":4,"num_gnn_layers":3}'

out_json="smoke-slurm/pet-xl-${SLURM_JOB_PARTITION}-${SLURM_JOB_ID}.json"
out_log="smoke-slurm/pet-xl-${SLURM_JOB_PARTITION}-${SLURM_JOB_ID}.log"

.venv/bin/python scripts/run_cell.py \
  --worktree worktrees/everything \
  --python /home/boittier/metawork/.venv/bin/python \
  --dataset /home/boittier/metawork/metatrain/tests/resources/qm9_reduced_100.xyz \
  --key U0 --num-workers 0 --batch-size 8 --device cuda --epochs 2 \
  --timeout-s 1900 \
  --variant everything --dataset-name qm9-xl --repeat 0 \
  --model-hypers-json "$XL_HYPERS" \
  --out "$out_json" --log "$out_log"
status=$?

echo
echo "== model size / param count / memory (from the benchmark log) =="
grep -E "^model (hypers|parameters):|peak_|memory" "$out_log" || true

echo
echo "job finished at $(date) with status $status"
exit $status
