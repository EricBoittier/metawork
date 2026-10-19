#!/usr/bin/env bash
# The structures where mlip.cpp and metatrain differ most (roundtrip-check
# 175691): adaptive radii, kept edges, per-atom energies, and per element two
# isolated atoms (composition offset + isolated-atom output) on both sides.
# Then the 175691 comparison again with --save (predictions.npz for chemiscope).
#   SRC=$(madcore/snapshot-mlipcpp.sh) sbatch --export=ALL,SRC madcore/cutoff-diag.sbatch
#SBATCH --job-name=cutoff-diag
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
RT=${RT:-$PROJ/runs/roundtrip-check-175691}
MODEL=${MODEL:-$PROJ/runs/mlipcpp-step-2pow24-big6adapt16-muon10-145880/model.gguf}
DUMP=${DUMP:-$PROJ/runs/mlipcpp-eval-atoms-146241/big24-145880.test.bin}
TEST=/home/boittier/data/madcore/memmap-2pow24/test
IDS=${IDS:-"162636 103022 47112 107003 2496"}
OUT=$PROJ/runs/cutoff-diag-$SLURM_JOB_ID
VPY=/home/boittier/metawork/.venv-blackwell/bin/python
export PYTHONPATH=/home/boittier/metawork/metatrain/main-baseline/src
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GGML_CUDA_NO_TF32=1
mkdir -p $OUT/dimers
echo "host $(hostname)  src $SRC ($(cat $SRC/HEAD))  out $OUT"
module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGGML_NATIVE=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > $OUT/cmake.log 2>&1 || { tail -40 $OUT/cmake.log; exit 1; }
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-md > $OUT/build.log 2>&1 || { grep error $OUT/build.log | head -30; exit 1; }
MD=$BUILD/bin/mlipcpp-md
for i in $IDS; do
    $MD edges $MODEL --data $TEST --index $i > $OUT/$i.txt 2> $OUT/$i.err
    grep -E "^E engine" $OUT/$i.txt
done
# every element of the test split: two atoms 20 A apart in vacuum
$VPY - $TEST/a.bin $OUT/dimers <<'PY'
import sys, numpy as np, ase, ase.io
for z in np.unique(np.fromfile(sys.argv[1], np.int32)):
    ase.io.write(f"{sys.argv[2]}/{z}.xyz", ase.Atoms(numbers=[z, z], positions=[[0, 0, 0], [20, 0, 0]]))
PY
for f in $OUT/dimers/*.xyz; do
    z=$(basename $f .xyz)
    $MD edges $MODEL --xyz $OUT/dimers/$z.xyz > $OUT/dimers/$z.txt 2> $OUT/dimers/$z.err || echo "Z $z: edges failed ($(tail -1 $OUT/dimers/$z.err))"
done
$VPY -W ignore /home/boittier/metawork/madcore/cutoff_diag.py --ckpt $RT/model.ckpt --split $TEST --edges-dir $OUT \
    --dimers $OUT/dimers $IDS || echo "!! cutoff_diag failed"
echo "=== 175691 comparison again, saving every prediction"
$VPY -W ignore /home/boittier/metawork/madcore/roundtrip_compare.py compare --ckpt $RT/model.ckpt --pt $RT/model.pt \
    --split $TEST --mlipcpp $RT/mlipcpp --mlipcpp-src $SRC --dump $DUMP --save $OUT/predictions.npz | tail -3
echo done
