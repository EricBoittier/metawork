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

rm -rf metatomic/python/metatomic_core/build metatomic/python/metatomic_torch/build metatensor/python/metatensor_torch/build /tmp/metatensor-build-* 2>/dev/null

# metatensor-torch's build backend pins its own dependency to whatever torch
# is *currently installed* at build time, so torch must already be at the
# right version (not just constrained) before this runs.
uv pip install --python .venv/bin/python "torch==2.5.1+cu121" \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install --python .venv/bin/python \
  --constraint pipeline-bench/slurm/torch-constraints.txt \
  --no-build-isolation \
  -e "./metatensor[torch]" -e "./metatomic[torch]" \
  ase psutil numpy omegaconf jsonschema "pydantic>=2.12" typing_extensions \
  python-hostlist tqdm huggingface_hub vesin
status=$?

echo "job finished at $(date) with status $status"
exit $status
