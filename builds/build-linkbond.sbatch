#!/usr/bin/env bash
# Build both GROMACS lines with the cut-bond fix and run their metatomic and topology unit tests
# (local helper, not tracked). Free build QOS, no GPU.
#   libtorch: ~/metawork/gromacs-torch-linkbond (oniom-torch-linkbond)  -> builds/gromacs-torch-linkbond-kuma-x86_64
#   C API:    ~/metawork/gromacs-capi-binding   (capi-binding)          -> builds/gromacs-capi-linkbond-kuma-x86_64
#SBATCH --job-name=gmx-linkbond-build
#SBATCH --partition=h100
#SBATCH --qos=build
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/build-linkbond-%j.out
set -euo pipefail
module purge
module load gcc/13.2.0 cuda/12.4.1
M=/home/boittier/metawork
V=$M/builds/torch-cu126/lib/python3.12/site-packages
export PATH=$M/builds/tools/bin:$PATH CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export LD_LIBRARY_PATH=$V/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}
export ONIOM_PY=$M/builds/torch-cu126/bin/python JOBS=16
PYARGS="-DPython_ROOT_DIR=/home/boittier/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu -DPython_FIND_VIRTUALENV=STANDARD -DCUDA_cudart_LIBRARY=$V/nvidia/cuda_runtime/lib/libcudart.so.12 -DBUILD_TESTING=ON"
echo "== $(hostname) $(date -Is)"

build() {  # build NAME SRC EXTRA_CMAKE_ARGS
    echo "== $1 $(git -C $2 rev-parse --short HEAD) $(date -Is)"
    GMX_SRC=$2 GMX=$M/builds/gromacs-$1-kuma-x86_64/bin/gmx GMX_CMAKE_ARGS="$PYARGS $3" $M/etc/oniom/build-gromacs.sh
    cmake --build $2/build-kuma-x86_64 -j 16 --target mta_applied_forces-test topology-test
    $2/build-kuma-x86_64/bin/mta_applied_forces-test
    $2/build-kuma-x86_64/bin/topology-test --gtest_filter='*Embedded*:*Link*'
    "$M/builds/gromacs-$1-kuma-x86_64/bin/gmx" --version | grep -E 'GROMACS version|GPU support'
}
build capi-linkbond $M/gromacs-capi-binding "-DCMAKE_PREFIX_PATH=$M/builds/capi/core-prefix;$V/metatensor/lib/cmake"
build torch-linkbond $M/gromacs-torch-linkbond ""
echo "== done $(date -Is)"
