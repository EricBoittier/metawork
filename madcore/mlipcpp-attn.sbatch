#!/usr/bin/env bash
# mlip.cpp fused attention kernels: accuracy of every variant against a
# double-precision reference, and per-call times on PET shapes (mlipcpp-attn-test)
#SBATCH --job-name=mlipcpp-attn
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail
SRC=${SRC:-/home/boittier/metawork/mlip.cpp}
PROJ=/work/cosmo/boittier/lyra/mlipcpp
BUILD=${BUILD:-$PROJ/build-$(basename $SRC)}
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg
echo "host $(hostname)  src $SRC"
nvidia-smi --query-gpu=name,driver_version --format=csv
module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGGML_NATIVE=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > /dev/null
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-attn-test 2>&1 | grep -E "error|warning: .*(spill|pet-attn)|ptxas" | head -40 || true
[ -x $BUILD/bin/mlipcpp-attn-test ] || { echo "!! build failed"; cmake --build $BUILD --target mlipcpp-attn-test 2>&1 | tail -40; exit 1; }
$BUILD/bin/mlipcpp-attn-test ${REPS:-20}
