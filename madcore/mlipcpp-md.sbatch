#!/usr/bin/env bash
# mlip.cpp GPU MD (mlipcpp-md): build, validate the engine against labels, the
# CPU GraphModel, the trainer's per-structure errors and finite differences,
# then short NVE runs (conservative / RESPA / direct) on a crystal and a molecule.
#SBATCH --job-name=mlipcpp-md
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
set -euo pipefail
SRC=${SRC:?snapshot from madcore/snapshot-mlipcpp.sh}
PROJ=/work/cosmo/boittier/lyra/mlipcpp
BUILD=${BUILD:-$PROJ/build-$(basename $SRC)}
OUT=$PROJ/runs/mlipcpp-md-$SLURM_JOB_ID
DATA=/home/boittier/data/madcore/memmap-2pow24/test
DUMPS=$PROJ/runs/mlipcpp-eval-atoms-146241
BIG=$PROJ/runs/mlipcpp-step-2pow24-big6adapt16-muon10-145880/model.gguf
BASE=$PROJ/runs/mlipcpp-step-2pow24-muon10-145207/model.gguf
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p $OUT
echo "host $(hostname)  src $SRC  out $OUT"
nvidia-smi --query-gpu=name,driver_version --format=csv
module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGGML_NATIVE=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > /dev/null
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-md mlipcpp-train 2>&1 | grep -E "error|(md|mlipcpp_md)[^ ]*:[0-9]+:[0-9]+: warning" | head -60 || true
for t in mlipcpp-md mlipcpp-train; do
    [ -x $BUILD/bin/$t ] || { echo "!! build of $t failed"; cmake --build $BUILD --target $t 2>&1 | tail -60; exit 1; }
done
MD=$BUILD/bin/mlipcpp-md

if [ "${STAGE:-all}" = edges ]; then
export GGML_CUDA_NO_TF32=1
for i in 0 559 155961 35217; do
    for m in base:$BASE big:$BIG; do
        echo "=== $i ${m%%:*}: engine inputs"
        $MD edges ${m#*:} --data $DATA --index $i | grep -E "^E engine|^forward|^GraphModel"
    done
done
exit 0
fi
if [ "${STAGE:-all}" = diag2 ]; then
export GGML_CUDA_NO_TF32=1
for m in big:$BIG base:$BASE; do
    tag=${m%%:*} model=${m#*:}
    echo "=== path $tag crys128 / mol28"
    $MD path $model --data $DATA --index 55621 --repeat 2,2,2 --length 3 --points 1500 > $OUT/path-$tag-crys128.tsv
    $MD path $model --data $DATA --index 157219 --length 1.5 --points 1500 > $OUT/path-$tag-mol28.tsv
    $MD check $model $DATA --n 300 --fd 0 --csv $OUT/check-$tag.csv \
        --dump $DUMPS/${tag}24-$( [ $tag = big ] && echo 145880 || echo 145207 ).test.bin | grep -E "vs |structures"
done
ls -la $OUT
exit 0
fi
if [ "${STAGE:-all}" = diag ]; then
export GGML_CUDA_NO_TF32=1
for pad in 0 4 32; do
    echo "=== padding $pad (FP32): big 145880"
    MLIPCPP_MD_PAD=$pad $MD check $BIG $DATA --n 300 --fd 0 --dump $DUMPS/big24-145880.test.bin
done
echo "=== finite differences (FP32), max |F| <= 5 eV/A: big 145880, base 145207"
$MD check $BIG $DATA --n 3000 --fd 8 --fd-steps 1e-2,3e-3,1e-3,3e-4 | grep -E "finite|structures"
$MD check $BASE $DATA --n 3000 --fd 8 --fd-steps 1e-2,3e-3,1e-3,3e-4 | grep -E "finite|structures"
STEPS=2000
for dt in 0.5 0.25; do
  for tag in "mol28 $BIG --data $DATA --index 157219" "crys128-base $BASE --data $DATA --index 55621 --repeat 2,2,2"; do
    set -- $tag
    echo "=== FP32 conservative NVE $1 dt $dt"
    $MD run $2 --integrator conservative --steps $(python3 -c "print(int($STEPS*0.5/$dt))") --dt $dt \
        --temperature 300 --seed 1 --log-every 100 "${@:3}" | awk 'NR<=3 || NR%4==0 || /graph builds/'
  done
done
exit 0
fi
if [ "${STAGE:-all}" != md ]; then
echo "=== check: big adaptive 145880"
$MD check $BIG $DATA --n ${NCHECK:-300} --fd 6 --dump $DUMPS/big24-145880.test.bin
echo "=== check: big adaptive 145880, FP32 GEMMs (no TF32)"
GGML_CUDA_NO_TF32=1 $MD check $BIG $DATA --n ${NCHECK:-300} --fd 6 --dump $DUMPS/big24-145880.test.bin
echo "=== check: big adaptive 145880, unfused attention (A/B of the bias gradient)"
MLIPCPP_NO_FUSED_ATTN=1 $MD check $BIG $DATA --n ${NCHECK:-300} --fd 6 --dump $DUMPS/big24-145880.test.bin
echo "=== check: base fixed cutoff 145207"
$MD check $BASE $DATA --n ${NCHECK:-300} --fd 6 --dump $DUMPS/base24-145207.test.bin
fi

STEPS=${STEPS:-2000}
run() { # tag model integrator system-args...
    local tag=$1 model=$2 integ=$3; shift 3
    echo "=== run $tag: $integ"
    $MD run $model --integrator $integ --steps $STEPS --dt 0.5 --temperature 300 --seed 1 --log-every 50 \
        --thermo $OUT/$tag-$integ.tsv --traj $OUT/$tag-$integ.xyz --traj-every 100 "$@" | \
        awk 'NR<=4 || /graph builds/ || NR%8==0'
}
for integ in conservative respa direct; do
    run crys128 $BIG $integ --data $DATA --index 55621 --repeat 2,2,2
    run mol28 $BIG $integ --data $DATA --index 157219
done
run crys128-base $BASE conservative --data $DATA --index 55621 --repeat 2,2,2
echo done
