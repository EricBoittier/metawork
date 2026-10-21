#!/bin/bash
# Evaluate mlipcpp-train conservative fine-tunes (runs/mlipcpp-train-cons-JOB/model.gguf):
#   1. mlipcpp-md (built from SRC, padding fix 5c7a81a) on each GGUF: the ft-check protocol
#      (2^20 val 2000 rows, 2^24 test 300 rows + finite differences, 1 ps NVE crys128 / mol28,
#      conservative + RESPA); 147625's GGUF too, as the same-binary reference
#   2. GGUF -> metatrain ckpt (gguf_to_checkpoint.py) -> metatomic .pt
#   3. ft_eval.py (metatrain, independent of mlip.cpp) on the 175773 structures
#      (2^24 test every 84th, 2^20 val every 5th) with base 145880 and 147625 alongside
#   SRC=$(madcore/snapshot-mlipcpp.sh) RUNS="176442 175917" sbatch --export=ALL madcore/ft-eval-mlipcpp.sbatch
# MD_ENVS: padding modes for step 1, "-" = default (-34.5 bias), "MLIPCPP_PAD_EXACT=1"
# = padded keys -inf (models trained that way); metatrain (step 3) always has the finite bias.
# REF=0 skips 147625 in step 1.
# STAGES: subset of "check md convert eval" (default all). TRAJ_EVERY=N (inner steps, 0 = off)
# writes NVE trajectories as extxyz (positions + velocities), zstd-compressed to *.xyz.zst.
#SBATCH --job-name=ft-eval-mlipcpp
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -uo pipefail
SRC=${SRC:?SRC: a source snapshot (madcore/snapshot-mlipcpp.sh)}
RUNS=${RUNS:-"176442 175917"}  # space-separated (sbatch --export splits on commas)
STAGES=${STAGES:-"check md convert eval"}; TRAJ_EVERY=${TRAJ_EVERY:-0}
has() { [[ " $STAGES " == *" $1 "* ]]; }
PROJ=/work/cosmo/boittier/lyra/mlipcpp; D=/home/boittier/data/madcore
BUILD=$PROJ/build-$(basename $SRC)
OUT=$PROJ/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}; mkdir -p $OUT; cd $OUT
VPY=/home/boittier/metawork/.venv-blackwell/bin/python; MTT=$(dirname $VPY)/mtt
INIT=$PROJ/runs/mlipcpp-bench-adapt16-145278/init.pt
S=/work/cosmo/boittier/lyra/mlipcpp/transfer-20261019
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg TORCH_HOME=$C/torch
export PYTHONPATH=/home/boittier/metawork/metatrain/main-baseline/src  # c3bc0cf2, as the fine-tunes
export VESIN_CUDA_MAX_PAIRS_PER_POINT=8192
echo "host $(hostname)  src $SRC ($(cut -c1-8 $SRC/HEAD)$( [ -s $SRC/DIFF.patch ] && echo +dirty))  runs $RUNS"

module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9 nccl/2.28.7-1-cuda
t=$(date +%s)
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > cmake.log || { tail -30 cmake.log; exit 1; }
cmake --build $BUILD -j $SLURM_CPUS_PER_TASK --target mlipcpp-md > build.log 2>&1 || { grep -E "error" build.log | head -40; exit 1; }
echo "build: $(( $(date +%s) - t )) s"

status=0
declare -A GGUF=([147625]=$PROJ/finetune/cons-ft-train-147625/model.gguf)
for j in $RUNS; do GGUF[$j]=$PROJ/runs/mlipcpp-train-cons-$j/model.gguf; done

echo "=== 1. mlipcpp-md checks + NVE"
for j in $RUNS $( [[ ${REF:-1} == 1 ]] && echo 147625 ); do
  for e in ${MD_ENVS:--}; do
    [[ $e == - ]] && e=
    m=${GGUF[$j]}; o=$OUT/md-$j${e:+-exact}; mkdir -p $o
    echo "##### $j${e:+  $e}  $m"
    MD="env $e $BUILD/bin/mlipcpp-md"
    if has check; then
    $MD check $m $D/memmap-2pow20/val --n 2000 --max-atoms 100000 --fd 0 | grep -E "structures|vs labels" || status=1
    $MD check $m $D/memmap-2pow24/test --n 300 --fd 6 --csv $o/check-test.csv | grep -E "structures|vs |finite" || status=1
    fi
    has md || continue
    for integ in conservative respa; do
        for sys in "crys128 --index 55621 --repeat 2,2,2" "mol28 --index 157219"; do
            set -- $sys; tag=$1; shift
            echo "--- $tag $integ"
            traj=(); (( TRAJ_EVERY > 0 )) && traj=(--traj $o/$tag-$integ.xyz --traj-every $TRAJ_EVERY)
            $MD run $m --integrator $integ --steps 2000 --dt 0.5 --temperature 300 --seed 1 --log-every 50 \
                --data $D/memmap-2pow24/test "$@" --thermo $o/$tag-$integ.tsv "${traj[@]}" > $o/$tag-$integ.log 2>&1 || status=1
            grep -iE "graph builds|error" $o/$tag-$integ.log; head -1 $o/$tag-$integ.tsv; tail -1 $o/$tag-$integ.tsv
            [[ -s $o/$tag-$integ.xyz ]] && zstd -q --rm -f -10 $o/$tag-$integ.xyz && ls -lh $o/$tag-$integ.xyz.zst | cut -d" " -f5,9
        done
    done
  done
done

echo "=== 2. GGUF -> ckpt -> .pt"
MODELS="$S/pet-big24-145880.pt $S/pet-big24-consft-147625.pt"
for j in $( has convert && echo $RUNS ); do
    $VPY -W ignore $SRC/scripts/train/gguf_to_checkpoint.py --init $INIT --gguf ${GGUF[$j]} -o $OUT/cons-$j \
        > convert-$j.log 2>&1 || { echo "convert $j FAILED"; tail -20 convert-$j.log; status=1; continue; }
    $MTT export $OUT/cons-$j.ckpt -o $OUT/cons-$j.pt > export-$j.log 2>&1 \
        || { echo "export $j FAILED"; tail -20 export-$j.log; status=1; continue; }
    MODELS="$MODELS $OUT/cons-$j.pt"
done

echo "=== 3. ft_eval.py (metatrain): $MODELS"
if has eval; then
    $VPY -W ignore /home/boittier/metawork/madcore/ft_eval.py --refit --out report.json \
        --split test24=$D/memmap-2pow24/test:84 --split val20=$D/memmap-2pow20/val:5 \
        $MODELS || status=1
fi
echo "done (status $status)"
exit $status
