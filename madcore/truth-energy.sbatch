#!/usr/bin/env bash
#SBATCH --job-name=truth-energy
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail
export PYTHONPATH=/home/boittier/metawork/metatrain/main-baseline/src
/home/boittier/metawork/.venv-blackwell/bin/python -W ignore /home/boittier/metawork/madcore/truth_energy.py \
    /work/cosmo/boittier/lyra/mlipcpp/finetune/cons-ft-smoke-147598/start.ckpt \
    /home/boittier/data/madcore/memmap-2pow24/test 0 559 155961 35217 2>&1 | grep -v -i warn
