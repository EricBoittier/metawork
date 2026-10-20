#!/usr/bin/env bash
# Conservative fine-tune in mlip.cpp (branch train/pet-conservative): -dE/dx
# force loss plus the direct head's, from INIT (default: the metatrain
# conservative fine-tune 147625, as 175824), with its own baselines. Per-
# structure training errors of every epoch (--log-errors, before each update:
# the last epoch's are the label-quality signal) and val/test errors
# (--dump-errors).
#   pilot:  SRC=$(madcore/snapshot-mlipcpp.sh) STEPS=300 GPUS=1 sbatch --qos=debug --time=01:00:00 \
#               --gpus-per-node=1 --cpus-per-task=32 --mem=180G madcore/mlipcpp-train-cons.sbatch
#   full:   SRC=... GPUS=8 sbatch --gpus-per-node=8 --cpus-per-task=96 --mem=480G madcore/mlipcpp-train-cons.sbatch
# MLIPCPP_PAD_EXACT=1 in the environment: padded attention keys get -inf, not
# the finite -34.5 bias, so gradients no longer depend on the batch layout
# (176621 / 176630); evaluate such models with MLIPCPP_PAD_EXACT=1 as well.
#SBATCH --job-name=mlipcpp-train-cons
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=480G
#SBATCH --time=10:00:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail

SRC=${SRC:?SRC: a source snapshot (madcore/snapshot-mlipcpp.sh)}
PROJ=/work/cosmo/boittier/lyra/mlipcpp
BUILD=${BUILD:-$PROJ/build-$(basename $SRC)}
OUT=$PROJ/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}
DATA=${DATA:-/home/boittier/data/madcore/memmap-2pow24}
INIT=${INIT:-$PROJ/finetune/cons-ft-train-147625/model.gguf}
GPUS=${GPUS:-8}
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg
mkdir -p "$OUT"
echo "host $(hostname)  src $SRC ($(cat $SRC/HEAD | cut -c1-8)$( [ -s $SRC/DIFF.patch ] && echo +dirty))  init $INIT  pad ${MLIPCPP_PAD_EXACT:+exact}${MLIPCPP_PAD_EXACT:-1e-15}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9 nccl/2.28.7-1-cuda
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGGML_NATIVE=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > $OUT/cmake.log || { tail -30 $OUT/cmake.log; exit 1; }
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-train > $OUT/build.log 2>&1 \
    || { grep -E "error" $OUT/build.log | head -30; exit 1; }
LOCAL=$TMPDIR/$(basename $DATA)
cp -rL $DATA $LOCAL
( while sleep 30; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; done ) > $OUT/gpumem.log &
trap "kill $! 2>/dev/null || true" EXIT

$BUILD/bin/mlipcpp-train --data $LOCAL --init $INIT -o $OUT/model.gguf --gpus $GPUS \
    --batch-size ${BATCH:-32} --epochs ${EPOCHS:-3} --max-steps ${STEPS:--1} --lr ${LR:-2e-5} \
    --optimizer ${OPT:-adam} --muon-lr-scale ${MUON_SCALE:-1} --log-every ${LOG_EVERY:-500} \
    --threads $SLURM_CPUS_PER_TASK --cache-dir $C/mlipcpp-nl --model-baselines \
    --conservative --energy-weight ${EW:-0.1} --force-weight ${FW:-5} --nc-force-weight ${NCW:-1} \
    --log-errors $OUT/train-errors.bin --log-errors-epochs ${EPOCHS:-3} --dump-errors $OUT/errors
echo "peak GPU MiB $(sort -n $OUT/gpumem.log | tail -1)"
