#!/usr/bin/env bash
# Build metatensor[torch]/metatomic[torch] on a compute node instead of the
# login node, which caps every user's cgroup at 8GiB -- not enough headroom
# for several parallel PyTorch-header C++ compiles at once.
#SBATCH --job-name=metawork-build
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --chdir=/home/boittier/metawork
#SBATCH --output=/home/boittier/metawork/pipeline-bench/smoke-slurm/%x-%j.out
#SBATCH --error=/home/boittier/metawork/pipeline-bench/smoke-slurm/%x-%j.err

set -uo pipefail

echo "== job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK cpus, mem cap: =="
cat /sys/fs/cgroup/memory.max 2>/dev/null || true

module load nvhpc/24.7-mpi
module load cuda/12.5.1
export CC=gcc CXX=g++ CUDAHOSTCXX=g++
export CMAKE_BUILD_PARALLEL_LEVEL=$SLURM_CPUS_PER_TASK
export CARGO_BUILD_JOBS=$SLURM_CPUS_PER_TASK
export MAKEFLAGS="-j$SLURM_CPUS_PER_TASK"
source "$HOME/.cargo/env"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
which uv

rm -rf metatomic/python/metatomic_core/build metatomic/python/metatomic_torch/build \
  metatensor/python/metatensor_core/build metatensor/python/metatensor_learn/build \
  metatensor/python/metatensor_operations/build metatensor/python/metatensor_torch/build \
  /tmp/metatensor-build-* 2>/dev/null

# metatensor-torch's build backend pins its own dependency to whatever torch
# is *currently installed* at build time, so torch must already be at the
# right version (not just constrained) before this runs.
uv pip install --python .venv/bin/python "torch==2.5.1+cu121" \
  --index-url https://download.pytorch.org/whl/cu121

# metatensor-torch's CMake (find_package(metatensor)) needs metatensor-core
# already installed in the venv's site-packages at configure time, and
# metatomic-torch's build imports metatensor.torch -- so each _core package
# has to be installed on its own, as a completed step, before the [torch]
# extra that depends on it is built. Bundling core+torch in one uv
# invocation lets uv build torch's wheel against the not-yet-installed (or
# still-stale) core wheel from the same batch.
uv pip install --python .venv/bin/python \
  --constraint pipeline-bench/slurm/torch-constraints.txt \
  --no-build-isolation \
  --reinstall-package metatensor-core --refresh-package metatensor-core \
  "./metatensor/python/metatensor_core"
status=$?

if [ $status -eq 0 ]; then
  uv pip install --python .venv/bin/python \
    --constraint pipeline-bench/slurm/torch-constraints.txt \
    --no-build-isolation \
    --reinstall-package metatensor --reinstall-package metatensor-torch \
    --refresh-package metatensor --refresh-package metatensor-torch \
    -e "./metatensor[torch]"
  status=$?
fi

if [ $status -eq 0 ]; then
  uv pip install --python .venv/bin/python \
    --constraint pipeline-bench/slurm/torch-constraints.txt \
    --no-build-isolation \
    --reinstall-package metatomic-core --refresh-package metatomic-core \
    "./metatomic/python/metatomic_core"
  status=$?
fi

if [ $status -eq 0 ]; then
  # metatomic-torch depends on metatensor-torch, which is a local path (URL)
  # dependency -- uv refuses to resolve a transitive URL dependency unless it
  # is also a direct requirement/constraint of *this* invocation, even though
  # it is already installed and satisfies the pin. Re-listing metatensor[torch]
  # here does not rebuild it (no --reinstall/--refresh-package for it).
  uv pip install --python .venv/bin/python \
    --constraint pipeline-bench/slurm/torch-constraints.txt \
    --no-build-isolation \
    --reinstall-package metatomic --reinstall-package metatomic-torch \
    --refresh-package metatomic --refresh-package metatomic-torch \
    -e "./metatensor[torch]" -e "./metatomic[torch]" \
    ase psutil numpy omegaconf jsonschema "pydantic>=2.12" typing_extensions \
    python-hostlist tqdm huggingface_hub vesin
  status=$?
fi

echo "job finished at $(date) with status $status"
exit $status
