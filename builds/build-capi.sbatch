#!/usr/bin/env bash
# Build the C-API stack on kuma (local helper, not tracked): metatomic-core (libmetatomic), the
# metatomic-torch plugin, and GROMACS from the capi-binding branch, single precision with CUDA
# non-bondeds. CPU-only build QOS (free); nvcc compiles for the H100/L40S architectures.
#   metatomic: ~/metawork/metatomic-torch-plugin (fork torch-plugin = metatomic-core + plugin)
#   GROMACS:   ~/metawork/gromacs-capi-binding   (fork capi-binding)
#SBATCH --job-name=gmx-capi-build
#SBATCH --partition=h100
#SBATCH --qos=build
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/build-capi-%j.out
set -euo pipefail
module purge
module load gcc/13.2.0 cuda/12.4.1
M=/home/boittier/metawork
B=$M/builds/capi
MTA=$M/metatomic-torch-plugin
V=$M/builds/torch-cu126/lib/python3.12/site-packages
export PATH=$M/builds/tools/bin:$HOME/.cargo/bin:$PATH CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export CARGO_NET_OFFLINE=true   # crates fetched on the login node
echo "== $(hostname) $(date -Is) metatomic $(git -C $MTA rev-parse --short HEAD) gromacs $(git -C $M/gromacs-capi-binding rev-parse --short HEAD)"
cmake --version | head -1; nvcc --version | tail -1; cargo --version

cmake -S $MTA/metatomic-core -B $B/build-core -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$V/metatensor/lib/cmake" -DCMAKE_INSTALL_PREFIX=$B/core-prefix
cmake --build $B/build-core -j 16 --target install

cmake -S $MTA/metatomic-torch-plugin -B $B/build-plugin -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$V/torch/share/cmake;$V/metatensor/lib/cmake;$V/metatensor_torch/torch-2.13/lib/cmake;$V/metatomic/torch/torch-2.13/lib/cmake;$B/core-prefix" \
    -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" -DCMAKE_CUDA_ARCHITECTURES="89;90" -DTORCH_CUDA_ARCH_LIST="8.9;9.0" \
    -DCUDA_cudart_LIBRARY=$V/nvidia/cuda_runtime/lib/libcudart.so.12
cmake --build $B/build-plugin -j 16
ls -l $B/build-plugin/libmetatomic_torch_plugin.so

# GROMACS: build-gromacs.sh with the capi-binding checkout, finding libmetatomic and metatensor
export ONIOM_PY=$M/builds/torch-cu126/bin/python JOBS=16 GMX_SRC=$M/gromacs-capi-binding
export GMX=$M/builds/gromacs-capi-kuma-x86_64/bin/gmx
export GMX_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$B/core-prefix;$V/metatensor/lib/cmake -DPython_ROOT_DIR=/home/boittier/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu -DPython_FIND_VIRTUALENV=STANDARD -DCUDA_cudart_LIBRARY=$V/nvidia/cuda_runtime/lib/libcudart.so.12"
export LD_LIBRARY_PATH=$V/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}
$M/etc/oniom/build-gromacs.sh
"$GMX" --version | grep -E 'GROMACS version|Precision|GPU support|SIMD'
ldd "$GMX" | grep -E 'metatomic|metatensor|torch' || true
echo "== done $(date -Is)"
