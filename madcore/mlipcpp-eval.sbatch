#!/usr/bin/env bash
# mlip.cpp: per-structure val/test errors of trained checkpoints
# (mlipcpp-train --epochs 0 --dump-errors), all on the same DATA memmap.
# MODELS: one "label path.gguf [extra flags]" per line (e.g. --max-attention 5e7
# for the big models). Outputs: $OUT/<label>.{val,test}.bin (SystemError records).
#SBATCH --job-name=mlipcpp-eval
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=00:45:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail

SRC=${SRC:-/home/boittier/metawork/mlip.cpp} # a snapshot path: build exactly the submitted source
PROJ=/work/cosmo/boittier/lyra/mlipcpp
BUILD=${BUILD:-$PROJ/build-$(basename $SRC)}
OUT=$PROJ/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}
DATA=${DATA:-/home/boittier/data/madcore/memmap-2pow24}
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg
mkdir -p "$OUT"
[ -n "${MODELS:-}" ] || { echo "set MODELS"; exit 1; }
echo "host $(hostname)  src $SRC  data $DATA"
module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGGML_NATIVE=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > $OUT/cmake.log
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-train > $OUT/build.log 2>&1 \
    || { tail -40 $OUT/build.log; exit 1; }
# val/ and test/ only (eval-only skips train/): copy them to node-local disk
LOCAL=$TMPDIR/$(basename $DATA)
mkdir -p $LOCAL && for s in val test; do cp -rL $DATA/$s $LOCAL/; done
ln -sfn $(readlink -f $DATA/train) $LOCAL/train # opened (mapped), never read
while read -r label path flags; do
    [ -z "$label" ] && continue
    echo "== $label: $path $flags"
    $BUILD/bin/mlipcpp-train --data $LOCAL --init $path -o $OUT/unused.gguf --epochs 0 --gpus 1 \
        --threads $SLURM_CPUS_PER_TASK --cache-dir $C/mlipcpp-nl --dump-errors $OUT/$label $flags \
        || echo "!! $label failed"
done <<< "$MODELS"
ls -la $OUT
