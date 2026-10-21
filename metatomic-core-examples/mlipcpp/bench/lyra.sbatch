#!/usr/bin/env bash
# GPU head to head on lyra (SCITAS): build the plugin with CUDA, check it
# against PyTorch, then time ggml-cuda (direct, plugin C API, plugin Python)
# against metatomic-torch on CUDA, plus both on the node's CPUs.
#
#   cd metatomic-core-examples/mlipcpp && sbatch bench/lyra.sbatch <model.gguf> <model.pt>
#
# Needs the clones, build/venv and metatomic-core made by build.sh beforehand,
# on a login node without modules: cargo can not link under lyra's gcc module.
#SBATCH --job-name=mta-mlipcpp-bench
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail

GGUF=$1
PT=$2
E=${E:-$SLURM_SUBMIT_DIR}
B=$E/build
VPY=${TORCH_PYTHON:-/home/boittier/metawork/.venv-blackwell/bin/python}
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm PIP_CACHE_DIR=$C/pip XDG_CACHE_HOME=$C/xdg
OUT=${OUT:-$PWD/bench-$SLURM_JOB_ID.csv}

module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9
echo "host $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

JOBS=$SLURM_CPUS_PER_TASK PYTHON=$VPY \
  MLIPCPP_CMAKE_ARGS="-DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=100;120 -DGGML_NATIVE=OFF" \
  "$E/build.sh" > /dev/null
echo "mlip.cpp $(git -C "$B/mlip.cpp" rev-parse --short HEAD), metatomic $(git -C "$B/metatomic" rev-parse --short HEAD)"

MTS="$("$B/venv/bin/python" -c 'import metatensor, os; print(os.path.dirname(metatensor.__file__))')"
L=$B/mlipcpp-build/lib
"${CXX:-c++}" -O2 -std=c++20 -I"$B/prefix/include" -I"$MTS/include" -I"$B/mlip.cpp/include" "$E/bench/bench.cpp" -o "$B/bench" \
  "$L/libmlipcpp.a" "$L/libggml.a" $(ls "$L"/libggml-*.a | grep -v base) "$L/libggml-base.a" "$L/libfmt.a" \
  -fopenmp -L"$(dirname "$(dirname "$(which nvcc)")")/lib64" -lcudart -lcublas -lcublasLt -lcuda \
  -L"$B/prefix/lib" -L"$MTS/lib" -lmetatomic -lmetatensor \
  -Wl,--disable-new-dtags -Wl,-rpath,"$B/prefix/lib" -Wl,-rpath,"$MTS/lib"

P=$B/mlipcpp-metatomic.so
echo "== correctness: plugin on CUDA vs finite differences, and vs PyTorch"
"$B/metatomic_plugin" "$P" "$GGUF" '{"backend": "cuda"}' | tail -4
TORCH_PYTHON=$VPY "$E/compare-torch.sh" "$GGUF" "$PT"

echo "== timings -> $OUT"
GPU_CELLS="1 2 3 4 5 6"
CPU_CELLS="1 2 3 4"
echo "wrapper,backend,threads,n_atoms,mode,ms" > "$OUT"
MLIPCPP_BACKEND=cuda "$B/bench" "$P" "$GGUF" $GPU_CELLS >> "$OUT"
MLIPCPP_BACKEND=cuda "$B/venv/bin/python" "$E/bench/bench.py" plugin "$P" "$GGUF" $GPU_CELLS >> "$OUT"
BENCH_DEVICE=cuda "$VPY" -W ignore "$E/bench/bench.py" torch "$PT" $GPU_CELLS 2> /dev/null | grep -v "^Warp" >> "$OUT"
MLIPCPP_NUM_THREADS=$SLURM_CPUS_PER_TASK "$B/bench" "$P" "$GGUF" $CPU_CELLS >> "$OUT"
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK "$VPY" -W ignore "$E/bench/bench.py" torch "$PT" $CPU_CELLS 2> /dev/null \
  | grep -v "^Warp" >> "$OUT"
"$B/venv/bin/python" "$E/bench/summarize.py" "$OUT"
