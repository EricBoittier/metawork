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

# --- train: 2 GPUs (one worker thread each), batch 64 per GPU ---
# INIT: init.gguf of a mlipcpp-dev run (PET defaults), or OPTIONS: a metatrain
# yaml whose initial PET is built and exported here (e.g. a larger model)
[ -n "${INIT:-}${OPTIONS:-}" ] || { echo "set INIT or OPTIONS"; exit 1; }
module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGGML_NATIVE=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > $OUT/cmake.log
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-train mlipcpp-train-check > $OUT/build.log 2>&1 \
    || { tail -40 $OUT/build.log; exit 1; }
LOCAL=$TMPDIR/$(basename $DATA)
cp -rL $DATA $LOCAL # -L: DATA may be a symlink to /work
NL="--cache-dir $C/mlipcpp-nl"

if [ -z "${INIT:-}" ]; then
    mkdir -p $OUT/init && (cd $OUT/init && for d in train val test; do ln -sfn $LOCAL/$d $d; done &&
        PYTHONPATH=$MTT_SRC SLURM_NTASKS=1 $VPY $SRC/scripts/train/make_init.py $OPTIONS -o $OUT/init.pt) \
        || { echo "!! make_init failed"; exit 1; }
    $VPY $SRC/scripts/export_pytorch/export_pet_gguf.py --model $OUT/init.pt --nc-forces -o $OUT/init.gguf \
        > $OUT/export.log 2>&1 || { tail -40 $OUT/export.log; exit 1; }
    tail -3 $OUT/export.log
    INIT=$OUT/init.gguf
fi

echo "== gate"
$BUILD/bin/mlipcpp-train-check --data $LOCAL --init $INIT --out $OUT/check --n ${GATE_N:-64} --gpus 1 $NL --tol ${GATE_TOL:-1e-3} || echo "!! gate: check failed"
PYTHONPATH=$MTT_SRC $VPY $SRC/scripts/train/check_gradients.py --init $(dirname $INIT)/init.pt --data $LOCAL --dump $OUT/check --tol ${GATE_TOL:-1e-3} || echo "!! gate: gradients vs PyTorch failed"

# AB_ENV: space-separated VAR=1 switches (A=1+B=1 sets both; not a comma: sbatch --export splits on it) (e.g. MLIPCPP_NO_FUSED_ATTN=1
# GGML_CUDA_NO_TF32=1); profile and every throughput config are repeated with
# each one set, labelled "-<VAR>"
for E in "" ${AB_ENV:-}; do
echo "== profile${E:+ ($E)}"
env ${E//+/ } $BUILD/bin/mlipcpp-train-check --data $LOCAL --init $INIT --out $OUT/profile --n ${GATE_N:-64} --gpus 1 $NL \
    --profile 1 --tol ${GATE_TOL:-1e-3} --sweep $([ -z "$E" ] && echo ${SWEEP:-0} || echo 0) || echo "!! profile failed"
done
# CPU_GATE=1: the same gate on the CPU backend (CPU reference kernels), 4 structures
if [ -n "${CPU_GATE:-}" ]; then
    echo "== CPU gate"
    $BUILD/bin/mlipcpp-train-check --data $LOCAL --init $INIT --out $OUT/check-cpu --n 4 --gpus 0 $NL --tol ${GATE_TOL:-1e-3} || echo "!! CPU gate: check failed"
    PYTHONPATH=$MTT_SRC $VPY $SRC/scripts/train/check_gradients.py --init $(dirname $INIT)/init.pt --data $LOCAL --dump $OUT/check-cpu --tol ${GATE_TOL:-1e-3} || echo "!! CPU gate: gradients vs PyTorch failed"
fi
# MUON_CHECK=1: the gate again with --gpus 2, which adds the Muon split check
# (split step == MLIPCPP_MUON_ONE_DEVICE step on random gradients, + ms per step)
if [ -n "${MUON_CHECK:-}" ]; then
    echo "== Muon split check (2 GPUs)"
    $BUILD/bin/mlipcpp-train-check --data $LOCAL --init $INIT --out $OUT/check-2gpu --n ${GATE_N:-64} --gpus 2 $NL --tol ${GATE_TOL:-1e-3} || echo "!! Muon split check failed"
fi
# COMPARE_INIT: memory/FLOP sweep and profile of another model (e.g. the 4.5 A base PET)
if [ -n "${COMPARE_INIT:-}" ]; then
    echo "== compare: profile + sweep of $COMPARE_INIT"
    $BUILD/bin/mlipcpp-train-check --data $LOCAL --init $COMPARE_INIT --out $OUT/compare --n 64 --gpus 1 $NL \
        --profile 1 --sweep 1 || echo "!! compare failed"
fi
[ -n "${SKIP_TRAJ:-}" ] || {

echo "== trajectory (30 steps, compare with the canary of 145112)"
$BUILD/bin/mlipcpp-train --data $LOCAL --init $INIT -o $OUT/traj.gguf --gpus 2 --batch-size 64 \
    --epochs 1 --max-steps 30 --log-every 2 --threads $SLURM_CPUS_PER_TASK $NL ${EXTRA:-} --pack worker || echo "!! trajectory failed"
echo "== trajectory, tiles (same steps: padding must not change the losses)"
$BUILD/bin/mlipcpp-train --data $LOCAL --init $INIT -o $OUT/traj-tile.gguf --gpus 2 --batch-size 64 \
    --epochs 1 --max-steps 30 --log-every 2 --threads $SLURM_CPUS_PER_TASK $NL ${EXTRA:-} --tile-nb-ratio 1.25 --pack worker || echo "!! trajectory failed"
echo "== trajectory, step packing (full-step mean: close to, not equal to, the DDP mean)"
$BUILD/bin/mlipcpp-train --data $LOCAL --init $INIT -o $OUT/traj-step.gguf --gpus 2 --batch-size 64 \
    --epochs 1 --max-steps 30 --log-every 2 --threads $SLURM_CPUS_PER_TASK $NL ${EXTRA:-} --pack step || echo "!! trajectory failed"

}

# throughput: one "label flags..." per line; struct/s is cumulative over the epoch
CONFIGS=${CONFIGS:-"step --pack step
step-muon10 --pack step --optimizer muon --muon-lr-scale 10"}
STEPS=${STEPS:-600}
# REF_BUILD: also run every config with that (older) build, for an A/B in one job
for B in $BUILD ${REF_BUILD:-}; do
for E in "" ${AB_ENV:-}; do
tag=$([ "$B" = "$BUILD" ] && echo "" || echo "ref-")
while read -r label flags; do
    [ -z "$label" ] && continue
    label=$tag$label${E:+-$(echo $E | tr "+=" "_-")}
    echo "== throughput $label ($STEPS steps, $(basename $B)${E:+, $E}): $flags"
    env ${E//+/ } $B/bin/mlipcpp-train --data $LOCAL --init $INIT -o $OUT/tput-$label.gguf --gpus 2 --batch-size 64 \
        --epochs 10 --max-steps $STEPS --log-every 100 --threads $SLURM_CPUS_PER_TASK $NL ${EXTRA:-} $flags \
        || echo "!! $label failed"
done <<< "$CONFIGS"
done
done
