#!/usr/bin/env bash
# Controlled learning curves on MAD-CORE: base PET (Muon x10) trained on a
# subset of the 2^24 train split (make_subset_memmap.py: KIND fps|random, N
# structures, optional GROUP), EPOCHS chosen for a fixed number of structures
# seen; val = 2^20 val (cheap per epoch), test = 2^24 test for every run.
# Writes the final model, per-structure test errors (--dump-errors) and the
# first epoch's online errors (--log-errors, prequential) to $OUT; per-epoch
# checkpoints stay on node-local disk. BUILD_ONLY=1: build and exit.
#SBATCH --job-name=mlipcpp-subset
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=01:30:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail

SRC=${SRC:?set SRC to a source snapshot}
VPY=/home/boittier/metawork/.venv-blackwell/bin/python
HERE=/home/boittier/metawork/madcore
PROJ=/work/cosmo/boittier/lyra/mlipcpp
BUILD=${BUILD:-$PROJ/build-$(basename $SRC)}
D=/home/boittier/data/madcore
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg
module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9
if [ -n "${BUILD_ONLY:-}" ]; then
    cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DGGML_NATIVE=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > $BUILD.cmake.log
    cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-train > $BUILD.log 2>&1 || { tail -40 $BUILD.log; exit 1; }
    echo "built $BUILD"; exit 0
fi

: ${KIND:?} ${N:?} ${EPOCHS:?}
OUT=$PROJ/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}
mkdir -p $OUT
echo "host $(hostname)  src $SRC  subset $KIND n=$N ${GROUP:+group=$GROUP }epochs=$EPOCHS"
# random picks read the whole 2^24 train split: copy it to node-local disk first
SRCMM=$D/memmap-2pow24
if [ "$KIND" = random ]; then
    mkdir -p $TMPDIR/src && cp -rL $D/memmap-2pow24/train $TMPDIR/src/train && SRCMM=$TMPDIR/src
fi
SUB=$TMPDIR/subset
$VPY -I $HERE/make_subset_memmap.py --src $SRCMM --kind $KIND --n $N ${GROUP:+--group $GROUP} \
    --val $D/memmap-2pow20/val --test $D/memmap-2pow24/test -o $SUB
cp $SUB/rows.npy $OUT/
rm -rf $TMPDIR/src
mkdir -p $TMPDIR/ckpt
t=$(date +%s)
$BUILD/bin/mlipcpp-train --data $SUB --init ${INIT:-$PROJ/runs/mlipcpp-dev-145099/init.gguf} -o $TMPDIR/ckpt/model.gguf \
    --gpus 2 --batch-size 64 --epochs $EPOCHS --threads $SLURM_CPUS_PER_TASK --log-every 200 \
    --optimizer muon --muon-lr-scale 10 --log-errors $OUT/online.bin --dump-errors $OUT/errors ${EXTRA:-}
echo "elapsed $(( $(date +%s) - t )) s"
cp $TMPDIR/ckpt/model.gguf $OUT/
ls -la $OUT
