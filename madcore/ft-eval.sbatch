#!/bin/bash
#SBATCH --job-name=ft-eval
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -eo pipefail
# Base 145880 and the conservative fine-tunes, as exported metatomic models, on identical structures
# (ft_eval.py): 2^24 test every 84th, 2^20 val every 5th, split overlap: none (175770).
S=/work/cosmo/boittier/lyra/mlipcpp/transfer-20261019; D=/home/boittier/data/madcore
export PYTHONPATH=/home/boittier/metawork/metatrain/main-baseline/src  # c3bc0cf2, as the fine-tunes
export VESIN_CUDA_MAX_PAIRS_PER_POINT=8192  # dense 6 A cells exceed the default 512
VPY=/home/boittier/metawork/.venv-blackwell/bin/python
OUT=/work/cosmo/boittier/lyra/mlipcpp/runs/ft-eval-$SLURM_JOB_ID; mkdir -p $OUT; cd $OUT
echo "host $(hostname)  metatrain $(git -C $PYTHONPATH/.. rev-parse --short HEAD)"
MODELS="$S/pet-big24-145880.pt $S/pet-big24-consft-147625.pt $S/pet-big24-consft-ew10-148417.pt"
if [ -n "${FT_JOBS:-}" ]; then  # new fine-tunes: export their checkpoints and evaluate those instead
    MODELS=""; MTT=$(dirname $VPY)/mtt
    for j in ${FT_JOBS//,/ }; do
        F=/work/cosmo/boittier/lyra/mlipcpp/finetune/cons-ft-train-$j
        [ -f $F/model.ckpt ] || { echo "no $F/model.ckpt, skipped"; continue; }
        $MTT export $F/model.ckpt -o $OUT/consft-$j.pt > export-$j.log 2>&1 && MODELS="$MODELS $OUT/consft-$j.pt"
    done
else
    $VPY -W ignore /home/boittier/metawork/madcore/ckpt_baselines_diff.py $S/*.ckpt
fi
$VPY -W ignore /home/boittier/metawork/madcore/ft_eval.py ${REFIT---refit} --out report.json \
    --split test24=$D/memmap-2pow24/test:84 --split val20=$D/memmap-2pow20/val:5 \
    $MODELS
