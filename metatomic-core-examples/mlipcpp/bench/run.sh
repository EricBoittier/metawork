#!/usr/bin/env bash
# Head to head timings of one model through every wrapper, on CPU:
# mlipcpp::Predictor, the plugin via the C API and via Python, and the
# TorchScript export via metatomic-torch's ASE calculator.
#
#   bench/run.sh <model.gguf> <model.pt|model.ckpt> [out.csv]
#
# Needs build.sh first; TORCH_PYTHON as in compare-torch.sh. Runs are
# sequential so that they do not compete for cores.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$HERE/build"
TORCH_PYTHON=${TORCH_PYTHON:-$HERE/../../.venv-upet/bin/python}
GGUF=$1
MODEL=$2
OUT=${3:-bench.csv}
PLUGIN="$BUILD/mlipcpp-metatomic.so"
MTS="$("$BUILD/venv/bin/python" -c 'import metatensor, os; print(os.path.dirname(metatensor.__file__))')"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

if [[ "$MODEL" == *.ckpt ]]; then
  (cd "$WORK" && "$(dirname "$TORCH_PYTHON")/mtt" export "$MODEL" -o model.pt > export.log 2>&1)
  MODEL="$WORK/model.pt"
fi

L="$BUILD/mlipcpp-build/lib"
c++ -O2 -std=c++20 -I"$BUILD/prefix/include" -I"$MTS/include" -I"$BUILD/mlip.cpp/include" \
  "$HERE/bench/bench.cpp" -o "$BUILD/bench" \
  "$L/libmlipcpp.a" "$L/libggml.a" "$L/libggml-cpu.a" "$L/libggml-base.a" "$L/libfmt.a" -fopenmp \
  -L"$BUILD/prefix/lib" -L"$MTS/lib" -lmetatomic -lmetatensor \
  -Wl,--disable-new-dtags -Wl,-rpath,"$BUILD/prefix/lib" -Wl,-rpath,"$MTS/lib"

echo "wrapper,backend,threads,n_atoms,mode,ms" > "$OUT"
for threads in 4 12 24; do
  MLIPCPP_NUM_THREADS=$threads "$BUILD/bench" "$PLUGIN" "$GGUF" >> "$OUT"
done
MLIPCPP_NUM_THREADS=12 "$BUILD/venv/bin/python" "$HERE/bench/bench.py" plugin "$PLUGIN" "$GGUF" >> "$OUT"
for threads in 12 24; do
  OMP_NUM_THREADS=$threads "$TORCH_PYTHON" -W ignore "$HERE/bench/bench.py" torch "$MODEL" 2> /dev/null \
    | grep -v "^Warp" >> "$OUT"
done
"$BUILD/venv/bin/python" "$HERE/bench/summarize.py" "$OUT"
