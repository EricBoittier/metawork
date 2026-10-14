#!/usr/bin/env bash
# mlip.cpp training, dev tranche: CUDA build -> initial PET (same path as the
# metatrain reference run) -> GGUF export -> gradient parity vs PyTorch
# (the gate) -> 1-GPU training smoke test.
#SBATCH --job-name=mlipcpp-dev
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
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
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-train mlipcpp-train-check > $OUT/build.log 2>&1 \
    || { tail -40 $OUT/build.log; exit 1; }
echo "build: $(( $(date +%s) - t )) s"

# --- stage data on node-local NVMe ---
LOCAL=$TMPDIR/$(basename $DATA)
cp -rL $DATA $LOCAL # -L: DATA may be a symlink to /work

# --- initial PET exactly as `mtt train` builds it, then GGUF ---
t=$(date +%s)
if [ -n "${REUSE_INIT:-}" ]; then cp $REUSE_INIT/init.pt $OUT/init.pt; fi
if [ ! -f $OUT/init.pt ]; then
mkdir -p $OUT/init && cd $OUT/init
for d in train val test; do ln -sfn $LOCAL/$d $d; done
$VPY $SRC/scripts/train/make_init.py /home/boittier/metawork/madcore/options-pet-bs64.yaml -o $OUT/init.pt
fi
$VPY $SRC/scripts/export_pytorch/export_pet_gguf.py --model $OUT/init.pt --nc-forces -o $OUT/init.gguf > $OUT/export.log 2>&1 \
    || { tail -40 $OUT/export.log; exit 1; }
tail -3 $OUT/export.log
echo "init + export: $(( $(date +%s) - t )) s"

# --- gate: gradient parity against metatrain/PyTorch (float64) ---
for dev in 1 0; do
    $BUILD/bin/mlipcpp-train-check --data $LOCAL --init $OUT/init.gguf --out $OUT/check-gpu$dev --n 64 --gpus $dev --cache-dir $C/mlipcpp-nl \
        || { echo "STEP FAILED: check --gpus $dev"; continue; }
    $VPY $SRC/scripts/train/check_gradients.py --init $OUT/init.pt --data $LOCAL --dump $OUT/check-gpu$dev || true
done

# --- diagnostic: per-node gradients on a small CPU batch ---
$BUILD/bin/mlipcpp-train-check --data $LOCAL --init $OUT/init.gguf --out $OUT/nodes-cpu --n 8 --gpus 0 \
    --cache-dir $C/mlipcpp-nl --node-grads 1 \
    && $VPY $SRC/scripts/train/compare_node_grads.py --init $OUT/init.pt --dump $OUT/nodes-cpu \
    || echo "STEP FAILED: node gradients"

# --- smoke: 1 GPU, 300 steps, neighbour-list cache on /work ---
$BUILD/bin/mlipcpp-train --data $LOCAL --init $OUT/init.gguf -o $OUT/smoke.gguf --gpus 1 \
    --epochs 1 --max-steps ${SMOKE_STEPS:-30} --log-every 10 --threads $SLURM_CPUS_PER_TASK \
    --cache-dir $C/mlipcpp-nl || echo "STEP FAILED: smoke"
