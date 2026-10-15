#!/usr/bin/env bash
# mlip.cpp PET training on MAD-CORE 2^18, 2 GPUs, batch 64 per GPU: the
# mlip.cpp side of the comparison with train-pet-bs64-2gpu.sbatch (metatrain).
#SBATCH --job-name=mlipcpp-bs64-2gpu
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=03:00:00
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
# build here (ccache): the binary is then exactly this job's source state
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGGML_NATIVE=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > $OUT/cmake.log
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-train mlipcpp-train-check > $OUT/build.log 2>&1 \
    || { tail -40 $OUT/build.log; exit 1; }
LOCAL=$TMPDIR/$(basename $DATA)
cp -rL $DATA $LOCAL # -L: DATA may be a symlink to /work
# optional: the neighbour-list builder of this build must reproduce existing
# caches byte for byte (val/test of NLCHECK, rebuilt in a scratch cache dir)
if [ -n "${NLCHECK:-}" ]; then
    CHK=$TMPDIR/nlcheck && mkdir -p $CHK && cp -rL $NLCHECK $CHK/data
    $BUILD/bin/mlipcpp-train --data $CHK/data --init $INIT -o /dev/null --gpus 2 --epochs 0 \
        --threads $SLURM_CPUS_PER_TASK --cache-dir $CHK/nl > $OUT/nlcheck.log 2>&1
    for f in $CHK/nl/nl-*.bin; do
        cmp $f $C/mlipcpp-nl/$(basename $f) || { echo "!! neighbour lists differ: $(basename $f)"; exit 1; }
    done
    echo "neighbour-list builder reproduces $(ls $CHK/nl/nl-*.bin | wc -l) cached lists byte for byte"
    rm -rf $CHK
fi

# gate on this build (GATE_N structures and tolerance GATE_TOL: fp32 sums over
# thousands of neighbours at large cutoffs reach ~3e-3): gradients vs metatrain/PyTorch (float64) and
# micro-batching == full batch; refuse to train otherwise
$BUILD/bin/mlipcpp-train-check --data $LOCAL --init $INIT --out $OUT/check --n ${GATE_N:-64} --gpus 1 --cache-dir $C/mlipcpp-nl --tol ${GATE_TOL:-1e-3}
PYTHONPATH=$MTT_SRC $VPY $SRC/scripts/train/check_gradients.py --init $(dirname $INIT)/init.pt --data $LOCAL --dump $OUT/check --tol ${GATE_TOL:-1e-3}

# canary: 30 steps on both GPUs through the epoch-end path (val, checkpoint
# write), then reload the checkpoint in eval-only mode
$BUILD/bin/mlipcpp-train --data $LOCAL --init $INIT -o $OUT/canary.gguf --gpus 2 --batch-size 64 \
    --epochs 1 --max-steps 30 --threads $SLURM_CPUS_PER_TASK --cache-dir $C/mlipcpp-nl --log-every 2 ${EXTRA:-}
$BUILD/bin/mlipcpp-train --data $LOCAL --init $OUT/canary.gguf -o /dev/null --gpus 2 --epochs 0 \
    --threads $SLURM_CPUS_PER_TASK --cache-dir $C/mlipcpp-nl ${EXTRA:-}

( while sleep 5; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits; done ) > $OUT/gpumem.log &
sampler=$!
t=$(date +%s)
$BUILD/bin/mlipcpp-train --data $LOCAL --init $INIT -o $OUT/model.gguf --gpus 2 --batch-size 64 \
    --epochs ${EPOCHS:-10} --threads $SLURM_CPUS_PER_TASK --cache-dir $C/mlipcpp-nl --log-every 200 --mb-growth ${MB_GROWTH:-4} ${EXTRA:-}
echo "elapsed $(( $(date +%s) - t )) s, peak GPU MiB $(cut -d, -f1 $OUT/gpumem.log | sort -n | tail -1)"
kill $sampler
