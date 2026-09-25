#!/usr/bin/env bash
# Builds a SEPARATE venv (.venv-blackwell) for this cluster's Blackwell GPUs
# (b200 = sm_100, rtx6000 = sm_120). The shared /.venv/ is pinned to
# torch==2.5.1+cu121 (sm_90 max, see slurm/torch-constraints.txt) and is
# still used on kuma (L40S/H100, sm_89/sm_90) -- this does NOT touch it,
# on purpose, so kuma keeps working unchanged.
#
# Reuses etc/_torch_cuda.sh's auto-detecting torch/CUDA helpers (the same
# ones etc/setup-metawork.sh itself uses) instead of a hardcoded pin, so
# whatever torch build actually initializes on this driver is what gets
# installed -- and builds metatensor[torch]/metatomic[torch] from the
# already-cloned local checkouts against that torch, mirroring
# build-stack.sbatch's install order but into .venv-blackwell.
#
# Must run on an actual GPU compute node: ensure_torch_for_driver reads
# nvidia-smi, and the login node caps every user's cgroup at 8GiB (not
# enough for parallel C++ compiles -- see build-stack.sbatch).
#
# Submit per-partition:
#   sbatch --partition=rtx6000 --job-name=build-blackwell-venv-rtx6000 slurm/build-blackwell-venv.sbatch
#   sbatch --partition=b200 --job-name=build-blackwell-venv-b200 slurm/build-blackwell-venv.sbatch
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --chdir=/home/boittier/metawork
#SBATCH --output=/home/boittier/metawork/pipeline-bench/smoke-slurm/%x-%j.out
#SBATCH --error=/home/boittier/metawork/pipeline-bench/smoke-slurm/%x-%j.err

set -uo pipefail

echo "== job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), partition $SLURM_JOB_PARTITION =="
date
nvidia-smi -L || echo "nvidia-smi not available"
nvidia-smi --query-gpu=name,driver_version --format=csv || true

source "$HOME/.cargo/env" 2>/dev/null || true
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
which uv

# /usr/local/cuda-12.7 on this cluster is runtime-libs-only (no nvcc) --
# metatensor-torch/metatomic-torch compile actual CUDA kernels at build
# time and need a real nvcc. nvhpc/25.11-mpi bundles CUDA 13.0 (matching
# this driver's reported CUDA 13.0 exactly).
module load nvhpc/25.11-mpi
which nvcc && nvcc --version

# ensure_working_cuda_toolchain's dirname(dirname(nvcc)) heuristic assumes a
# standard toolkit layout ($CUDA_HOME/bin/nvcc, headers/libs right next to
# it) -- NVHPC splits the compiler driver (.../compilers/bin/nvcc) from the
# actual runtime headers/libs, which live in a SIBLING .../cuda/<ver>/ tree
# instead (confirmed: cmake's find_package(CUDA) failed with "missing
# CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY" while pointed at .../compilers).
# Point CUDA_HOME at the real one directly so both that heuristic and
# CMake's FindCUDA agree.
NVHPC_COMPILERS="$(dirname "$(dirname "$(command -v nvcc)")")"
NVHPC_CUDA_VER="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
export CUDA_HOME="$(dirname "$NVHPC_COMPILERS")/cuda/$NVHPC_CUDA_VER"
ls "$CUDA_HOME/include/cuda_runtime.h" "$CUDA_HOME/lib64/libcudart.so" >/dev/null
echo "CUDA_HOME=$CUDA_HOME"

# Loading the nvhpc module puts nvc++ first on PATH, which CMake then picks
# as the host C++ compiler instead of plain g++ -- confirmed the hard way:
# metatensor-torch failed to compile against it (PyTorch's compile-time
# get_fully_qualified_type_name() parses __PRETTY_FUNCTION__ with a
# hardcoded gcc/clang-specific prefix/suffix, which nvc++ doesn't match).
# nvcc itself still needs a compatible *host* compiler for .cu files,
# separate from CMake's own C/CXX compiler choice; both should be plain
# gcc/g++, matching build-stack.sbatch's CUDAHOSTCXX=g++ for kuma.
export CC=gcc CXX=g++ CUDAHOSTCXX=g++

VENV_DIR=/home/boittier/metawork/.venv-blackwell
VPY="$VENV_DIR/bin/python"

if [ ! -d "$VENV_DIR" ]; then
  echo "== creating $VENV_DIR =="
  uv venv "$VENV_DIR" --python 3.12
fi

# shellcheck source=etc/_torch_cuda.sh
source /home/boittier/metawork/etc/_torch_cuda.sh

echo
echo "== picking a torch build this driver can initialize =="
ensure_torch_for_driver "$VPY"
ensure_build_seed_packages "$VPY"
ensure_working_cuda_toolchain || true

echo
echo "== torch installed =="
"$VPY" -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
"

cd /home/boittier/metawork
status=0

# metatensor-torch/metatomic-torch's custom build backend pins their wheel's
# torch dependency to whatever it finds cached in these build/ dirs from a
# PREVIOUS build (confirmed the hard way: a first run of this script here
# silently re-pinned to the old torch==2.5.* left over from kuma's last
# build-stack.sbatch run against the shared /.venv/, discarding the fresh
# Blackwell-compatible torch just installed above). Safe to wipe: kuma's
# /.venv/ has its own fully-copied, independent install of these packages
# in its own site-packages (not an editable link back into this checkout),
# so clearing the shared source tree's build cache cannot affect it.
echo "== clearing stale build/ + egg-info caches (see comment above) =="
rm -rf metatomic/python/metatomic_core/build metatomic/python/metatomic_torch/build \
  metatomic/python/metatomic_ase/build \
  metatensor/python/metatensor_core/build metatensor/python/metatensor_learn/build \
  metatensor/python/metatensor_operations/build metatensor/python/metatensor_torch/build \
  metatensor/metatensor.egg-info metatomic/metatomic.egg-info \
  metatensor/python/metatensor_torch/metatensor_torch.egg-info \
  metatensor/python/metatensor_core/metatensor_core.egg-info \
  metatensor/python/metatensor_learn/metatensor_learn.egg-info \
  metatensor/python/metatensor_operations/metatensor_operations.egg-info \
  metatomic/python/metatomic_core/src/metatomic_core.egg-info \
  metatomic/python/metatomic_torch/src/metatomic_torch.egg-info \
  metatomic/python/metatomic_ase/src/metatomic_ase.egg-info \
  /tmp/metatensor-build-* 2>/dev/null

echo
echo "== metatensor-core =="
uv pip install --python "$VPY" --constraint "$VENV_DIR/.torch-constraint.txt" "${TORCH_EXCLUDE_NEWER_ARGS[@]}" \
  --no-build-isolation \
  --reinstall-package metatensor-core --refresh-package metatensor-core \
  "./metatensor/python/metatensor_core"
status=$?

# uv caches built-wheel metadata for a local path keyed by source content,
# not by which torch was installed when it was built -- confirmed the hard
# way: clearing the source tree's build/.egg-info (above) was not enough,
# uv still reused a cached metatensor-torch wheel declaring torch==2.5.*
# from the earlier (pre-fix) run, producing a hard, unsatisfiable conflict
# against this run's torch==2.14.0 constraint instead of silently
# downgrading. --reinstall-package/--refresh-package (matching
# build-stack.sbatch) force uv past that cache.
if [ $status -eq 0 ]; then
  echo
  echo "== metatensor[torch] =="
  uv pip install --python "$VPY" --constraint "$VENV_DIR/.torch-constraint.txt" "${TORCH_EXCLUDE_NEWER_ARGS[@]}" \
    --no-build-isolation \
    --reinstall-package metatensor --reinstall-package metatensor-torch \
    --refresh-package metatensor --refresh-package metatensor-torch \
    -e "./metatensor[torch]"
  status=$?
fi

if [ $status -eq 0 ]; then
  echo
  echo "== metatomic-core =="
  uv pip install --python "$VPY" --constraint "$VENV_DIR/.torch-constraint.txt" "${TORCH_EXCLUDE_NEWER_ARGS[@]}" \
    --no-build-isolation \
    --reinstall-package metatomic-core --refresh-package metatomic-core \
    "./metatomic/python/metatomic_core"
  status=$?
fi

if [ $status -eq 0 ]; then
  echo
  echo "== metatomic[torch] + plain deps =="
  uv pip install --python "$VPY" --constraint "$VENV_DIR/.torch-constraint.txt" "${TORCH_EXCLUDE_NEWER_ARGS[@]}" \
    --no-build-isolation \
    --reinstall-package metatomic --reinstall-package metatomic-torch \
    --refresh-package metatomic --refresh-package metatomic-torch \
    -e "./metatensor[torch]" -e "./metatomic[torch]" \
    ase psutil numpy omegaconf jsonschema "pydantic>=2.12" typing_extensions \
    python-hostlist tqdm huggingface_hub vesin
  status=$?
fi

echo
echo "== final sanity check: matmul on the actual GPU =="
if [ $status -eq 0 ]; then
  "$VPY" -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('device', torch.cuda.get_device_name(0))
a = torch.randn(1024, 1024, device='cuda')
b = torch.randn(1024, 1024, device='cuda')
c = a @ b
torch.cuda.synchronize()
print('matmul ok, result sum:', float(c.sum()))
"
  status=$?
fi

echo
echo "job finished at $(date) with status $status"
exit $status
