#!/bin/bash
# experimental.lorem (metatrain PR #1265, worktree metatrain/lorem-clean) on
# MAD-CORE level 18 in DiskDataset (.mta/.mts) format. See options-lorem-mts.yaml.
#SBATCH --job-name=lorem-madcore-2pow18
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --output=/home/boittier/data/madcore/lorem-runs/%x-%j.out
set -euo pipefail

# torch 2.5.1+cu121: kuma's driver (CUDA 12.6) is too old for .venv-blackwell's cu130
VENV=/home/boittier/metawork/.venv
DATA=/home/boittier/data/madcore/mts-2pow18-conservative
export PYTHONPATH=/home/boittier/metawork/metatrain/lorem-clean/src
# sphericart dlopens an unversioned libcudart.so; the venv only ships .so.12
export LD_LIBRARY_PATH=/home/boittier/data/madcore/lorem-runs/cudart:$VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# fold forces into energy.mts gradients; no-op for splits already converted
"$VENV/bin/python" /home/boittier/metawork/madcore/conservative_zip.py \
    /home/boittier/data/madcore/mts-2pow18 "$DATA" -j "$SLURM_CPUS_PER_TASK"

WORKDIR=/home/boittier/data/madcore/lorem-runs/$SLURM_JOB_ID
mkdir -p "$WORKDIR" && cd "$WORKDIR"
for split in train val test; do ln -sfn "$DATA/$split.zip" "$split.zip"; done

hostname; nvidia-smi -L
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), torch.__version__"
git -C "$PYTHONPATH/.." log -1 --oneline; git -C "$PYTHONPATH/.." diff --stat
"$VENV/bin/mtt" train /home/boittier/metawork/madcore/options-lorem-mts.yaml \
    -o lorem-madcore-2pow18.pt -e extensions "$@"
