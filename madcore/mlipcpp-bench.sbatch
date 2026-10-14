#!/usr/bin/env bash
# mlip.cpp speed work: gate + micro-batch equivalence, per-op GPU profile,
# 30-step 2-GPU trajectories (vs the canary of run 145112; dynamic and tiled) and
# throughput of the CONFIGS (default: step packing with Adam and Muon x10).
#SBATCH --job-name=mlipcpp-bench
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:45:00
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

mkdir -p "$OUT"

echo "host $(hostname)  mlip.cpp $(git -C $SRC rev-parse --short HEAD)$(git -C $SRC diff --quiet || echo +dirty)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# --- train: 2 GPUs (one worker thread each), batch 64 per GPU, PET defaults ---
INIT=${INIT:?set INIT to the init.gguf of a mlipcpp-dev run}
module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGGML_NATIVE=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > $OUT/cmake.log
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-train mlipcpp-train-check > $OUT/build.log 2>&1 \
    || { tail -40 $OUT/build.log; exit 1; }
LOCAL=$TMPDIR/$(basename $DATA)
cp -r $DATA $LOCAL
NL="--cache-dir $C/mlipcpp-nl"

echo "== gate"
$BUILD/bin/mlipcpp-train-check --data $LOCAL --init $INIT --out $OUT/check --n 64 --gpus 1 $NL || echo "!! gate: check failed"
PYTHONPATH=$MTT_SRC $VPY $SRC/scripts/train/check_gradients.py --init $(dirname $INIT)/init.pt --data $LOCAL --dump $OUT/check || echo "!! gate: gradients vs PyTorch failed"

echo "== profile"
$BUILD/bin/mlipcpp-train-check --data $LOCAL --init $INIT --out $OUT/profile --n 64 --gpus 1 $NL --profile 1 || echo "!! profile failed"

echo "== trajectory (30 steps, compare with the canary of 145112)"
$BUILD/bin/mlipcpp-train --data $LOCAL --init $INIT -o $OUT/traj.gguf --gpus 2 --batch-size 64 \
    --epochs 1 --max-steps 30 --log-every 2 --threads $SLURM_CPUS_PER_TASK $NL --pack worker
echo "== trajectory, tiles (same steps: padding must not change the losses)"
$BUILD/bin/mlipcpp-train --data $LOCAL --init $INIT -o $OUT/traj-tile.gguf --gpus 2 --batch-size 64 \
    --epochs 1 --max-steps 30 --log-every 2 --threads $SLURM_CPUS_PER_TASK $NL --tile-nb-ratio 1.25 --pack worker
echo "== trajectory, step packing (full-step mean: close to, not equal to, the DDP mean)"
$BUILD/bin/mlipcpp-train --data $LOCAL --init $INIT -o $OUT/traj-step.gguf --gpus 2 --batch-size 64 \
    --epochs 1 --max-steps 30 --log-every 2 --threads $SLURM_CPUS_PER_TASK $NL --pack step

# throughput: one "label flags..." per line; struct/s is cumulative over the epoch
CONFIGS=${CONFIGS:-"step --pack step
step-muon10 --pack step --optimizer muon --muon-lr-scale 10"}
STEPS=${STEPS:-600}
while read -r label flags; do
    [ -z "$label" ] && continue
    echo "== throughput $label ($STEPS steps): $flags"
    $BUILD/bin/mlipcpp-train --data $LOCAL --init $INIT -o $OUT/tput-$label.gguf --gpus 2 --batch-size 64 \
        --epochs 10 --max-steps $STEPS --log-every 100 --threads $SLURM_CPUS_PER_TASK $NL $flags \
        || echo "!! $label failed"
done <<< "$CONFIGS"
