#!/usr/bin/env bash
# pad_sensitivity.py: metatrain PET gradients, one batch vs two halves (float64, CPU)
#SBATCH --job-name=pad-sens
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
export PYTHONPATH=/home/boittier/metawork/metatrain/main-baseline/src
VPY=/home/boittier/metawork/.venv-blackwell/bin/python
P=/work/cosmo/boittier/lyra/mlipcpp
D=/home/boittier/data/madcore/memmap-2pow15
for m in $P/finetune/cons-ft-train-175775/start.ckpt $P/runs/mlipcpp-bench-adapt16-145278/init.pt; do
    echo "=== $m"
    $VPY -W ignore pad_sensitivity.py --model $m --data $D --n 16
    $VPY -W ignore pad_sensitivity.py --model $m --data $D --n 16 --conservative
done
