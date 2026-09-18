#!/usr/bin/env bash
# Plumbing-only GPU smoke test: Slurm allocation + CUDA visibility + the
# harness's worktree/run_cell scaffolding. NOT a real training run --
# metatrain's own import is expected to fail until the metatensor/metatomic
# metatomic-core branches are reconciled (see notes/ or ask Eric).
#
# Submit per-partition, e.g.:
#   sbatch --partition=l40s --job-name=pipeline-bench-smoke-l40s slurm/gpu-smoke.sbatch
#   sbatch --partition=h100 --job-name=pipeline-bench-smoke-h100 slurm/gpu-smoke.sbatch
#SBATCH --qos=debug
#SBATCH --time=00:30:00
# --gres=gpu:1 is rejected here ("CPU count per node can not be satisfied");
# this cluster's job_submit plugin wants --gpus instead.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --chdir=/home/boittier/metawork/pipeline-bench
#SBATCH --output=/home/boittier/metawork/pipeline-bench/smoke-slurm/%x-%j.out
#SBATCH --error=/home/boittier/metawork/pipeline-bench/smoke-slurm/%x-%j.err

set -uo pipefail

echo "== job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), partition $SLURM_JOB_PARTITION =="
date

echo
echo "== nvidia-smi -L =="
nvidia-smi -L || echo "nvidia-smi not available"

echo
echo "== torch/cuda probe in the run venv =="
/home/boittier/metawork/.venv/bin/python -c "
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
"

echo
echo "== snakemake smoke config (device=cuda) =="
.venv/bin/snakemake --profile profiles/local --config smoke=true --cores 1 --resources gpu=1 --rerun-incomplete
status=$?

echo
echo "== cell log tail (if any) =="
cell_log="results/cells/everything/qm9/w0_b8_cuda_r0.log"
if [ -f "$cell_log" ]; then
    tail -n 60 "$cell_log"
else
    echo "no cell log at $cell_log"
fi

echo
echo "job finished at $(date) with snakemake exit status $status"
exit $status
