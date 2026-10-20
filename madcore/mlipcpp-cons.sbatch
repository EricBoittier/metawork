#!/usr/bin/env bash
# mlip.cpp conservative training (branch train/pet-conservative), all gates on
# one source snapshot:
#   1. mlipcpp-jvp-check: tangents of E vs reverse mode, and reverse through
#      the tangent graph vs central differences (GPU, CPU; adaptive 145880 and
#      fixed-cutoff 145099)
#   2. mlipcpp-train-check --conservative 1 + check_gradients.py: loss and
#      parameter gradients vs metatrain/PyTorch float64 double backward
#      (adaptive init 145278, fixed init 145099; GPU, CPU), with micro-batch,
#      step-packing and tile consistency
#   3. a short mlipcpp-train --conservative run (Muon) from 145880
#   SRC=$(madcore/snapshot-mlipcpp.sh) sbatch madcore/mlipcpp-cons.sbatch
#SBATCH --job-name=mlipcpp-cons
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -uo pipefail

SRC=${SRC:?SRC: a source snapshot (madcore/snapshot-mlipcpp.sh)}
PROJ=/work/cosmo/boittier/lyra/mlipcpp
BUILD=${BUILD:-$PROJ/build-$(basename $SRC)}
OUT=$PROJ/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}
DATA=${DATA:-/home/boittier/data/madcore/memmap-2pow15}
STAGES=${STAGES:-"attn jvp grad train"}
VPY=/home/boittier/metawork/.venv-blackwell/bin/python
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg TORCH_HOME=$C/torch
export PYTHONPATH=/home/boittier/metawork/metatrain/main-baseline/src
BIG=$PROJ/runs/mlipcpp-step-2pow24-big6adapt16-muon10-145880/model.gguf
ADAPT=$PROJ/runs/mlipcpp-bench-adapt16-145278 # init.pt + init.gguf
FIXED=$PROJ/runs/mlipcpp-dev-145099           # init.pt + init.gguf
mkdir -p "$OUT"
echo "host $(hostname)  src $SRC ($(cat $SRC/HEAD | cut -c1-8)$( [ -s $SRC/DIFF.patch ] && echo +dirty))"

module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9 nccl/2.28.7-1-cuda
t=$(date +%s)
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=${BUILD_TYPE:-RelWithDebInfo} -DMLIPCPP_USE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > $OUT/cmake.log || { tail -30 $OUT/cmake.log; exit 1; }
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-jvp-check mlipcpp-train-check mlipcpp-train \
    > $OUT/build.log 2>&1 || { grep -E "error|Error" $OUT/build.log | head -40; exit 1; }
echo "build: $(( $(date +%s) - t )) s"

status=0
step() { echo "--- $*"; "$@" || { echo "STEP FAILED: $*"; status=1; }; }

if [[ $STAGES == *attn* ]]; then
    echo "=== fused attention tangent vs unfused (ggml_pet_attn_jvp)"
    step env GGML_CUDA_NO_TF32=1 $BUILD/bin/mlipcpp-jvp-check --attn-test 1 --gpus 1
fi

if [[ $STAGES == *jvp* ]]; then
    # JVP_ENV: extra env settings ('+'-joined) per run, "-" = none
    for model in ${JVP_MODELS:-$BIG $FIXED/init.gguf}; do
        for dev in ${JVP_DEVS:-1 0}; do
          for e in ${JVP_ENV:--}; do
            [[ $e == - ]] && e=
            echo "=== jvp-check $(basename $(dirname $model))  gpus $dev${e:+  env $e}"
            step env GGML_CUDA_NO_TF32=1 ${e//+/ } $BUILD/bin/mlipcpp-jvp-check --data $DATA --init $model --n ${N_JVP:-4} --gpus $dev \
                --cache-dir $C/mlipcpp-nl ${JVP_ARGS:-}
          done
        done
    done
fi

if [[ $STAGES == *grad* ]]; then
    W="--energy-weight ${EW:-0.5} --force-weight ${FW:-2} --nc-force-weight ${NCW:-0.25}"
    # GGUF:PyTorch pairs of the same weights; 145880 through its converted checkpoint
    for pair in ${GRAD_MODELS:-$ADAPT/init.gguf:$ADAPT/init.pt $FIXED/init.gguf:$FIXED/init.pt $BIG:$PROJ/finetune/cons-ft-train-175775/start.ckpt}; do
        gguf=${pair%%:*} pt=${pair#*:}
        for dev in ${GRAD_DEVS:-1 0}; do
          # GRAD_ENV: extra env settings ('+'-joined) per run, "-" = none
          for e in ${GRAD_ENV:--}; do
            [[ $e == - ]] && e=
            d=$OUT/grad-$(basename $(dirname $gguf))-gpu$dev${e:+-${e//[=+]/-}}
            echo "=== train-check --conservative 1 $(basename $(dirname $gguf))  gpus $dev${e:+  env $e}"
            step env GGML_CUDA_NO_TF32=1 ${e//+/ } $BUILD/bin/mlipcpp-train-check --data $DATA --init $gguf --out $d --n ${N_GRAD:-16} --first ${FIRST:-0} --gpus $dev \
                --cache-dir $C/mlipcpp-nl --conservative ${CONS:-1} $W --tile-nb-ratio 1.25
            step $VPY $SRC/scripts/train/check_gradients.py --init $pt --data $DATA --dump $d --tol ${TOL:-1e-3} ${CHECK_ARGS:-}
          done
        done
    done
fi

# profile: per-phase wall time and the node-by-node op profile of one step of
# PROF_N structures (2^24 split, the production fine-tune GGUF), conservative
# (FORCES F.* / MIXED M.fwd M.tan M.bwd) and the same batch non-conservative
if [[ $STAGES == *profile* ]]; then
    PINIT=${PINIT:-$PROJ/finetune/cons-ft-train-147625/model.gguf}
    # PROF_AB: extra conservative runs with these env settings ('+'-joined), for A/B
    for run in "1:" "0:" ${PROF_AB:+$(for e in $PROF_AB; do echo "1:$e"; done)}; do
        c=${run%%:*} e=${run#*:}
        echo "=== train-check --profile 1 --conservative $c  n ${PROF_N:-256}  $(basename $(dirname $PINIT))${e:+  env $e}"
        step env ${e//+/ } $BUILD/bin/mlipcpp-train-check --data ${PROF_DATA:-/home/boittier/data/madcore/memmap-2pow24} --init $PINIT \
            --out $OUT/profile-cons$c${e:+-${e//[=+]/-}} --n ${PROF_N:-256} --gpus 1 --cache-dir $C/mlipcpp-nl --profile 1 --tol 1 \
            --conservative $c --energy-weight 0.1 --force-weight 5 --nc-force-weight 1
    done
fi

# nodes: the exported FX graph in PyTorch on train-check's dumped batch, node
# by node (scripts/train/compare_node_grads.py; direct-force loss), for
# NODE_MODELS GGUF:checkpoint pairs: ggml vs the graph it interprets
if [[ $STAGES == *nodes* ]]; then
    for pair in ${NODE_MODELS:-$BIG:$PROJ/finetune/cons-ft-train-175775/start.ckpt}; do
        gguf=${pair%%:*} pt=${pair#*:}
        for dev in ${NODE_DEVS:-1 0}; do
            d=$OUT/nodes-$(basename $(dirname $gguf))-gpu$dev
            echo "=== node grads $(basename $(dirname $gguf))  gpus $dev"
            step env GGML_CUDA_NO_TF32=1 $BUILD/bin/mlipcpp-train-check --data $DATA --init $gguf --out $d --n ${N_NODES:-16} \
                --gpus $dev --cache-dir $C/mlipcpp-nl --node-grads 1
            step $VPY $SRC/scripts/train/compare_node_grads.py --init $pt --dump $d --tol ${TOL:-1e-3}
        done
    done
fi

if [[ $STAGES == *allreduce* ]]; then
    NG=$(nvidia-smi -L | wc -l)
    echo "=== train-check --gpus $NG: NCCL all-reduce + replicated steps vs the device-0 step"
    step env GGML_CUDA_NO_TF32=1 $BUILD/bin/mlipcpp-train-check --data $DATA --init $FIXED/init.gguf --out $OUT/allreduce \
        --n 16 --gpus $NG --cache-dir $C/mlipcpp-nl
fi

if [[ $STAGES == *train* ]]; then
    LOCAL=$TMPDIR/$(basename $DATA)
    cp -rL $DATA $LOCAL
    echo "=== 145880 as is (--epochs 0)"
    step $BUILD/bin/mlipcpp-train --data $LOCAL --init $BIG -o $OUT/none.gguf --gpus 1 --epochs 0 \
        --cache-dir $C/mlipcpp-nl --conservative
    for opt in ${OPTS:-"adam:1" "muon:1"}; do
        echo "=== mlipcpp-train --conservative (145880, ${opt%:*} lr scale ${opt#*:}, 2^15)"
        step $BUILD/bin/mlipcpp-train --data $LOCAL --init $BIG -o $OUT/cons-${opt%:*}.gguf --gpus 1 --batch-size 32 \
            --epochs 1 --max-steps ${STEPS:-300} --log-every 50 --lr 1e-5 --optimizer ${opt%:*} --muon-lr-scale ${opt#*:} \
            --cache-dir $C/mlipcpp-nl --model-baselines --conservative --energy-weight 0.1 --force-weight 5 --nc-force-weight 1
    done
fi
exit $status
