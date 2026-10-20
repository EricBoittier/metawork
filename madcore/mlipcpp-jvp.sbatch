#!/usr/bin/env bash
# mlip.cpp conservative-training checks (branch train/pet-conservative): CUDA
# build of a source snapshot, then mlipcpp-jvp-check (forward-mode tangents of
# the PET energy vs reverse mode, and reverse through the tangent graph vs
# central differences) on GPU and CPU, for the models in MODELS.
#   SRC=$(madcore/snapshot-mlipcpp.sh) sbatch madcore/mlipcpp-jvp.sbatch
#SBATCH --job-name=mlipcpp-jvp
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -uo pipefail

SRC=${SRC:?SRC: a source snapshot (madcore/snapshot-mlipcpp.sh)}
PROJ=/work/cosmo/boittier/lyra/mlipcpp
BUILD=${BUILD:-$PROJ/build-$(basename $SRC)}
OUT=$PROJ/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}
DATA=${DATA:-/home/boittier/data/madcore/memmap-2pow15}
N=${N:-4}
TARGETS=${TARGETS:-mlipcpp-jvp-check}
# adaptive big model (145880) and the fixed-cutoff base model (145099)
MODELS=${MODELS:-"$PROJ/runs/mlipcpp-step-2pow24-big6adapt16-muon10-145880/model.gguf $PROJ/runs/mlipcpp-dev-145099/init.gguf"}
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg
mkdir -p "$OUT"
echo "host $(hostname)  src $SRC ($(cat $SRC/HEAD | cut -c1-8)$( [ -s $SRC/DIFF.patch ] && echo +dirty))"

module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9
t=$(date +%s)
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=${BUILD_TYPE:-RelWithDebInfo} -DMLIPCPP_USE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > $OUT/cmake.log || { tail -30 $OUT/cmake.log; exit 1; }
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target $TARGETS > $OUT/build.log 2>&1 \
    || { grep -E "error|Error" $OUT/build.log | head -40; exit 1; }
echo "build: $(( $(date +%s) - t )) s"

status=0
for model in $MODELS; do
    for dev in 1 0; do
        echo "=== $(basename $(dirname $model))/$(basename $model)  gpus $dev"
        $BUILD/bin/mlipcpp-jvp-check --data $DATA --init $model --n $N --gpus $dev --cache-dir $C/mlipcpp-nl \
            ${JVP_ARGS:-} || status=1
    done
done
exit $status
