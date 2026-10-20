#!/usr/bin/env bash
# Throughput / memory probe for mlipcpp-train --conservative (the 175917
# recipe): one build, then short runs over CONFIGS on the 2^24 train split,
# scored on the small 2^15 val/test so evaluation stays negligible. Each config
# is "BATCH:MB_BUDGET[:TILE[:PACK[:GROWTH]]]" (space separated, set in the environment, not in
# --export, which splits on commas); a run takes max(MIN_STEPS, STRUCTS/BATCH)
# steps. GPU memory is sampled every second, per card.
#   SRC=$(madcore/snapshot-mlipcpp.sh) CONFIGS="32:5e7 512:5e7" sbatch madcore/mlipcpp-batch-probe.sbatch
#SBATCH --job-name=mlipcpp-batch-probe
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail

SRC=${SRC:?SRC: a source snapshot (madcore/snapshot-mlipcpp.sh)}
CONFIGS=${CONFIGS:?CONFIGS: "BATCH:MB_BUDGET[:TILE[:PACK[:GROWTH]]] ..."}
PROJ=/work/cosmo/boittier/lyra/mlipcpp
BUILD=${BUILD:-$PROJ/build-$(basename $SRC)}
OUT=$PROJ/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}
D=/home/boittier/data/madcore
INIT=${INIT:-$PROJ/finetune/cons-ft-train-147625/model.gguf}
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
STRUCTS=${STRUCTS:-120000}  # structures per config, all GPUs
MIN_STEPS=${MIN_STEPS:-30}
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg
mkdir -p "$OUT"
echo "host $(hostname)  src $SRC ($(cut -c1-8 $SRC/HEAD)$( [ -s $SRC/DIFF.patch ] && echo +dirty))  init $INIT  gpus $GPUS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9 nccl/2.28.7-1-cuda
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGGML_NATIVE=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > $OUT/cmake.log || { tail -30 $OUT/cmake.log; exit 1; }
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-train > $OUT/build.log 2>&1 \
    || { grep -E "error" $OUT/build.log | head -30; exit 1; }
LOCAL=$TMPDIR/probe
mkdir -p $LOCAL
cp -rL $D/memmap-2pow24/train $LOCAL/train
cp -rL $D/memmap-2pow15/val $LOCAL/val
cp -rL $D/memmap-2pow15/test $LOCAL/test

for cfg in $CONFIGS; do
    IFS=: read -r B M T P G <<< "$cfg"
    T=${T:-0} P=${P:-step} G=${G:-4}
    S=$(( STRUCTS / (B * GPUS) )); S=$(( S < MIN_STEPS ? MIN_STEPS : S ))
    R=$OUT/b$B-m$M-t$T-$P-g$G
    mkdir -p $R
    echo "=== batch $B/GPU  mb-budget $M  tile-nb-ratio $T  pack $P  mb-growth $G  steps $S"
    ( while sleep 1; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | paste -sd' '; done ) > $R/gpumem.log &
    MON=$!
    t0=$(date +%s)
    rc=0
    $BUILD/bin/mlipcpp-train --data $LOCAL --init $INIT -o $R/model.gguf --gpus $GPUS \
        --batch-size $B --mb-budget $M --tile-nb-ratio $T --pack $P --mb-growth $G --epochs 1 --max-steps $S --lr 2e-5 \
        --log-every $(( S / 6 > 0 ? S / 6 : 1 )) --threads $SLURM_CPUS_PER_TASK --cache-dir $C/mlipcpp-nl \
        --model-baselines --conservative --energy-weight 0.1 --force-weight 5 --nc-force-weight 1 \
        > $R/train.log 2>&1 || rc=$?
    kill $MON 2>/dev/null || true
    grep -E '^epoch 0 step|neighbour lists ready|epoch 0 done|error|out of memory' $R/train.log | tail -4
    # peak per card over the run (MiB)
    peak=$(awk '{for(i=1;i<=NF;i++){split($i,a,","); if(a[2]>m[a[1]])m[a[1]]=a[2]}} END{for(k in m)printf "%s:%s ",k,m[k]}' $R/gpumem.log)
    echo "rc $rc  wall $(( $(date +%s) - t0 )) s  peak MiB per card: $peak"
done
