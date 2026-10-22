#!/usr/bin/env bash
# Build both ONIOM GROMACS variants on a kuma h100 node (local helper, not tracked).
#SBATCH --job-name=gmx-eccgpu-build
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/build-eccgpu-%j.out
set -euo pipefail
module purge
module load gcc/13.2.0 cuda/12.4.1
export ONIOM_PY=/home/boittier/metawork/builds/torch-cu126/bin/python
export PATH=/home/boittier/metawork/builds/tools/bin:$PATH CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export JOBS=16 GMX_CMAKE_ARGS="-DPython_ROOT_DIR=/home/boittier/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu -DPython_FIND_VIRTUALENV=STANDARD -DCUDA_cudart_LIBRARY=/home/boittier/metawork/builds/torch-cu126/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
# torch cu126 needs the CUDA 12.6 runtime it ships (the 12.4 module's libcudart is too old)
export LD_LIBRARY_PATH=/home/boittier/metawork/builds/torch-cu126/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}
cd /home/boittier/metawork
echo "== $(hostname) $(date -Is)"; nvidia-smi -L; cmake --version | head -1; nvcc --version | tail -1
# the GPU embedded Coulomb correction, installed next to the running build (not over it)
export GMX=/home/boittier/metawork/builds/gromacs-oniom-kuma-x86_64-eccgpu/bin/gmx GMX_SRC=/home/boittier/metawork/gromacs-oniom-src
etc/oniom/build-gromacs.sh
"$GMX" --version | grep -E 'GROMACS version|Precision|GPU support'
echo "== done $(date -Is)"
