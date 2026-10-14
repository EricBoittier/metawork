#!/usr/bin/env bash
# mlip.cpp: per-node gradient diagnostic of the CPU backend (vs PyTorch), on
# the init of a dev run (REUSE_INIT). Formerly part of the dev tranche: CUDA build -> initial PET (same path as the
# metatrain reference run) -> GGUF export -> gradient parity vs PyTorch
# (the gate) -> 1-GPU training smoke test.
#SBATCH --job-name=mlipcpp-diag
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail

SRC=${SRC:-/home/boittier/metawork/mlip.cpp} # a snapshot path: build exactly the submitted source
MTT_SRC=/home/boittier/metawork/metatrain/main-baseline/src
VPY=/home/boittier/metawork/.venv-blackwell/bin/python
PROJ=/work/cosmo/boittier/lyra/mlipcpp
BUILD=${BUILD:-$PROJ/build-$(basename $SRC)} # one build dir per source snapshot
OUT=$PROJ/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}
DATA=${DATA:-/home/boittier/data/madcore/memmap-2pow18}
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg TORCH_HOME=$C/torch \
       TRITON_CACHE_DIR=$C/triton PIP_CACHE_DIR=$C/pip
export PYTHONPATH=$MTT_SRC SLURM_NTASKS=1 # make_init: never join a process group
mkdir -p "$OUT"

echo "host $(hostname)  mlip.cpp $(git -C $SRC rev-parse --short HEAD)$(git -C $SRC diff --quiet || echo +dirty)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# --- build (ccache + CPM cache on /work) ---
module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9
t=$(date +%s)
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=${BUILD_TYPE:-RelWithDebInfo} -DMLIPCPP_USE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > $OUT/cmake.log
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-train-check mlipcpp-op-test > $OUT/build.log 2>&1 \
    || { tail -40 $OUT/build.log; exit 1; }
echo "build: $(( $(date +%s) - t )) s"

# op-level: rms_norm * w backward on CPU and CUDA vs analytic
$BUILD/bin/mlipcpp-op-test || echo "OP TEST FAILED"

LOCAL=$TMPDIR/$(basename $DATA)
cp -rL $DATA $LOCAL # -L: DATA may be a symlink to /work
cp $REUSE_INIT/init.pt $OUT/init.pt
$VPY $SRC/scripts/export_pytorch/export_pet_gguf.py --model $OUT/init.pt --nc-forces -o $OUT/init.gguf > $OUT/export.log 2>&1
# CPU gate: full 64-structure batch vs PyTorch (float64)
$BUILD/bin/mlipcpp-train-check --data $LOCAL --init $OUT/init.gguf --out $OUT/check-cpu --n 64 --gpus 0 \
    --cache-dir $C/mlipcpp-nl
PYTHONPATH=$MTT_SRC $VPY $SRC/scripts/train/check_gradients.py --init $OUT/init.pt --data $LOCAL --dump $OUT/check-cpu \
    || echo "CPU GATE FAILED"

$BUILD/bin/mlipcpp-train-check --data $LOCAL --init $OUT/init.gguf --out $OUT/nodes-cpu --n 8 --gpus 0 \
    --cache-dir $C/mlipcpp-nl --node-grads 1
$VPY $SRC/scripts/train/compare_node_grads.py --init $OUT/init.pt --dump $OUT/nodes-cpu
