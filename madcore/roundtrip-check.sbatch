#!/usr/bin/env bash
# mlip.cpp-trained PET (default big24 145880) against metatrain / metatomic:
#   1. build the snapshot (mlipcpp-md, mlipcpp-train, the unit tests) and run the tests
#   2. GGUF -> metatrain checkpoint (gguf_to_checkpoint.py) -> metatomic .pt
#      (mtt export) and back to GGUF: every trained tensor must be identical
#   3. mlipcpp-md check on ~1000 test structures (all sizes, roundtrip_compare.py
#      pick): CUDA engine, GraphModel CPU and CUDA, the trainer's dump
#   4. roundtrip_compare.py: metatrain float64 / float32 and the metatomic .pt
#      against all of those, structure by structure
#   5. MAE and RMSE on the whole test split (CUDA engine, no references)
#   SRC=$(madcore/snapshot-mlipcpp.sh) sbatch --export=ALL,SRC madcore/roundtrip-check.sbatch
#SBATCH --job-name=roundtrip-check
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail
SRC=${SRC:?snapshot from madcore/snapshot-mlipcpp.sh}
PROJ=/work/cosmo/boittier/lyra/mlipcpp
BUILD=${BUILD:-$PROJ/build-$(basename $SRC)}
RUN=${RUN:-$PROJ/runs/mlipcpp-step-2pow24-big6adapt16-muon10-145880}
INIT=${INIT:-$PROJ/runs/mlipcpp-bench-adapt16-145278/init.pt}
DUMP=${DUMP:-$PROJ/runs/mlipcpp-eval-atoms-146241/big24-145880.test.bin}
TEST=/home/boittier/data/madcore/memmap-2pow24/test
OUT=$PROJ/runs/roundtrip-check-$SLURM_JOB_ID
VPY=/home/boittier/metawork/.venv-blackwell/bin/python
MTT=$(dirname $VPY)/mtt
export PYTHONPATH=/home/boittier/metawork/metatrain/main-baseline/src # c3bc0cf2, as the fine-tunes
CMP=/home/boittier/metawork/madcore/roundtrip_compare.py
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GGML_CUDA_NO_TF32=1
mkdir -p $OUT
cd $OUT
echo "host $(hostname)  src $SRC ($(cat $SRC/HEAD))  run $RUN  out $OUT  metatrain $(git -C $PYTHONPATH/.. rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version --format=csv

echo "=== 1. build + unit tests"
module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9
TESTS="test_graph_interpreter test_graph_model test_coulomb test_conservative test_neighbor_list"
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DMLIPCPP_BUILD_TESTS=ON -DGGML_NATIVE=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > cmake.log 2>&1 || { tail -40 cmake.log; exit 1; }
t0=$SECONDS
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target mlipcpp-md mlipcpp-train > build.log 2>&1 || true
for t in mlipcpp-md mlipcpp-train; do
    [ -x $BUILD/bin/$t ] || { echo "!! build of $t failed"; grep -E "error" build.log | head -40; exit 1; }
done
echo "build $((SECONDS - t0)) s"
cmake --build $BUILD -j "$SLURM_CPUS_PER_TASK" --target $TESTS > build-tests.log 2>&1 || grep -E "error" build-tests.log | head -20
for t in $TESTS; do # run from the build tree (some read fixtures relative to it)
    exe=$(find $BUILD -name $t -type f -executable | head -1)
    [ -n "$exe" ] || { echo "$t: not built"; continue; }
    (cd $BUILD/tests 2>/dev/null || cd $BUILD; timeout 600 $exe > $OUT/$t.log 2>&1) && r=ok || r="FAILED ($?)"
    echo "$t: $r  $(grep -E "^(All tests passed|test cases:)" $OUT/$t.log | tail -1)"
done
MD=$BUILD/bin/mlipcpp-md

echo "=== 2. GGUF -> checkpoint -> metatomic .pt, and back to GGUF"
$VPY -W ignore $SRC/scripts/train/gguf_to_checkpoint.py --init $INIT --gguf $RUN/model.gguf -o $OUT/model
$MTT export $OUT/model.ckpt -o $OUT/model.pt > export.log 2>&1 || { tail -20 export.log; exit 1; }
$VPY -W ignore $SRC/scripts/export_pytorch/export_pet_gguf.py --model $OUT/model.ckpt --nc-forces \
    -o $OUT/roundtrip.gguf > roundtrip.log 2>&1 || { tail -20 roundtrip.log; exit 1; }
$VPY - $RUN/model.gguf $OUT/roundtrip.gguf <<'EOF'
import json, sys
import numpy as np
sys.path.insert(0, "/home/boittier/metawork/mlip.cpp/scripts/train")
from gguf_to_checkpoint import baselines, read_gguf
(ma, ta), (mb, tb) = read_gguf(sys.argv[1]), read_gguf(sys.argv[2])
names = json.loads(ma["train.param_names"])
bad = [n for n in names if n not in tb or ta[n].tobytes() != tb[n].tobytes()]
print(f"round trip: {len(names) - len(bad)}/{len(names)} trained tensors byte-identical" + (f"; differ: {bad[:5]}" if bad else ""))
ba, bb = baselines(ma), baselines(mb)
print("baselines identical:", ba == bb)
EOF

echo "=== 3. mlipcpp-md check on the picked structures"
$VPY $CMP pick $TEST -o ids.txt --n 1000 --ends 10
$MD check $RUN/model.gguf $TEST --ids ids.txt --fd 0 --gpu-ref 1 --dump $DUMP --out $OUT/mlipcpp | \
    grep -E "structures|vs "

echo "=== 4. metatrain / metatomic comparison"
$VPY -W ignore $CMP compare --ckpt $OUT/model.ckpt --pt $OUT/model.pt --split $TEST --mlipcpp $OUT/mlipcpp \
    --mlipcpp-src $SRC --dump $DUMP -o $OUT/compare.json

echo "=== 5. whole test split: RMSE and MAE (CUDA engine)"
$MD check $RUN/model.gguf $TEST --n 100000000 --max-atoms 100000 --fd 0 --cpu 0 | grep -E "structures|vs labels"
ls -la $OUT
echo done
