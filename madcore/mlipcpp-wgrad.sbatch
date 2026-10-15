#!/usr/bin/env bash
# mlip.cpp backward copies: weight-gradient variants on PET shapes
# (mlipcpp-wgrad-test), then the per-op profile with the CONT/CPY breakdown
# by source (mlipcpp-train-check --profile) of INIT
#SBATCH --job-name=mlipcpp-wgrad
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail
SRC=${SRC:-/home/boittier/metawork/mlip.cpp}
PROJ=/work/cosmo/boittier/lyra/mlipcpp
BUILD=${BUILD:-$PROJ/build-$(basename $SRC)}
INIT=${INIT:-$PROJ/runs/mlipcpp-bench-bf16-145797/init.gguf} # d_pet 256 / 16 heads, 6 A
DATA=${DATA:-/home/boittier/data/madcore/memmap-2pow18}
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg
echo "host $(hostname)  src $SRC"
nvidia-smi --query-gpu=name,driver_version --format=csv
module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGGML_NATIVE=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > /dev/null
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-wgrad-test mlipcpp-train-check > $BUILD.log 2>&1 \
    || { tail -40 $BUILD.log; exit 1; }
LOCAL=$TMPDIR/$(basename $DATA)
cp -rL $DATA $LOCAL
for E in "" ${AB_ENV:-}; do
echo "== weight gradient + newton-schulz${E:+ ($E)}"
env ${E//+/ } $BUILD/bin/mlipcpp-wgrad-test ${REPS:-50}
echo "== profile${E:+ ($E)}"
env ${E//+/ } $BUILD/bin/mlipcpp-train-check --data $LOCAL --init $INIT --out $TMPDIR/profile --n 64 --gpus 1 \
    --cache-dir $C/mlipcpp-nl --profile 1 --tol 1e-3 || echo "!! profile failed"
done
