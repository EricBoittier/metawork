#!/usr/bin/env bash
# input_swap.py on the inputs dumped by 175704, attention epsilon 1e-15 and 1e-300
#SBATCH --job-name=input-swap-eps
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail
PROJ=/work/cosmo/boittier/lyra/mlipcpp
export PYTHONPATH=/home/boittier/metawork/metatrain/main-baseline/src OMP_NUM_THREADS=16
for eps in 1e-15 1e-300; do
    /home/boittier/metawork/.venv-blackwell/bin/python -W ignore /home/boittier/metawork/madcore/input_swap.py \
        --ckpt $PROJ/runs/roundtrip-check-175691/model.ckpt --dir $PROJ/runs/input-swap-175704 --eps $eps \
        162636 103022 47112 107003 2496 | grep -E "epsilon|===|E -"
done
