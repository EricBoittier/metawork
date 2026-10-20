#!/usr/bin/env bash
# cons_ft_diag.py on the smoke run's checkpoints (why the lr-0 gate failed)
#SBATCH --job-name=cons-ft-diag
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail
SMOKE=${SMOKE:?smoke run dir}
RUN=/work/cosmo/boittier/lyra/mlipcpp/runs/mlipcpp-step-2pow24-big6adapt16-muon10-145880
VAL=/home/boittier/data/madcore/memmap-2pow15/val
MD=/work/cosmo/boittier/lyra/mlipcpp/build-20261016-111128/bin/mlipcpp-md
export PYTHONPATH=/home/boittier/metawork/metatrain/main-baseline/src
cd $SMOKE
GGML_CUDA_NO_TF32=1 $MD check $RUN/model.gguf $VAL --n 300 --fd 0 --csv $SMOKE/val-mlipcpp.csv | grep -E "structures|vs labels"
/home/boittier/metawork/.venv-blackwell/bin/python -W ignore /home/boittier/metawork/madcore/cons_ft_diag.py \
    $RUN/model.gguf start.ckpt gate.ckpt $VAL $SMOKE/val-mlipcpp.csv 2>&1 | grep -v -i warn
