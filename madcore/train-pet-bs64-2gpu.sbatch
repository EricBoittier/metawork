#!/usr/bin/env bash
# Standard metatrain PET on MAD-CORE level 18, 2 GPUs, batch 64 per GPU:
# the reference side of the metatrain vs mlip.cpp training comparison.
# metatrain = fork origin/main (c3bc0cf2) worktree metatrain/main-baseline,
# without the perf/ddp-pet-fixes dipole-like force head.
#SBATCH --job-name=pet-bs64-2gpu
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=03:00:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail

VPY=/home/boittier/metawork/.venv-blackwell/bin/python
SRC=/home/boittier/metawork/metatrain/main-baseline/src
DATA=${DATA:-/home/boittier/data/madcore/memmap-2pow18}
WORK=/home/boittier/data/madcore/sweep/workdirs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}
export PYTHONPATH=$SRC
# caches off $HOME (SCITAS: $HOME is 100 GiB)
C=/work/cosmo/boittier/cache
export XDG_CACHE_HOME=$C/xdg TORCH_HOME=$C/torch TRITON_CACHE_DIR=$C/triton PIP_CACHE_DIR=$C/pip

echo "host $(hostname)  metatrain $(git -C "$SRC/.." rev-parse --short HEAD)  data $DATA"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$VPY" -c "import metatrain, torch; print(metatrain.__file__, torch.__version__)"

# stage the memmap (2 GB) on node-local NVMe so reads never touch /home
LOCAL=${TMPDIR:-/tmp}/$(basename "$DATA")
cp -r "$DATA" "$LOCAL"
mkdir -p "$WORK" && cd "$WORK"
for d in train val test; do ln -sfn "$LOCAL/$d" "$d"; done

( while sleep 5; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done ) > gpumem.log &
sampler=$!
start=$(date +%s)
srun --ntasks=2 "$(dirname "$VPY")/mtt" train /home/boittier/metawork/madcore/options-pet-bs64.yaml \
    -r "architecture.training.distributed_port=$((20000 + SLURM_JOB_ID % 20000))" \
    -o model.pt -e extensions
echo "elapsed $(( $(date +%s) - start )) s, peak GPU MiB $(sort -n gpumem.log | tail -1)"
kill $sampler
