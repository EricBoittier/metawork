#!/usr/bin/env bash
# padding_test.py: metatrain f64 with padded NEF slots as is / zero-filled
# (as mlip.cpp) / excluded from attention, vs mlip.cpp GraphModel CPU (175701).
#SBATCH --job-name=padding-test
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
set -euo pipefail
PROJ=/work/cosmo/boittier/lyra/mlipcpp
export PYTHONPATH=/home/boittier/metawork/metatrain/main-baseline/src
echo "host $(hostname)  metatrain $(git -C $PYTHONPATH/.. rev-parse --short HEAD)"
/home/boittier/metawork/.venv-blackwell/bin/python -W ignore /home/boittier/metawork/madcore/padding_test.py \
    --ckpt $PROJ/runs/roundtrip-check-175691/model.ckpt --predictions $PROJ/runs/cutoff-diag-175701/predictions.npz
echo done
