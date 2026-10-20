#!/usr/bin/env bash
# Configure-only check of the ggml source-cache key (patch contents, not
# paths): two snapshots of the same patches must share one patched copy.
#   SRC1=... SRC2=... sbatch madcore/cpm-key-test.sbatch
#SBATCH --job-name=cpm-key-test
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail
C=/work/cosmo/boittier/cache
export CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg
module purge
module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9
for s in $SRC1 $SRC2; do
    b=$TMPDIR/build-$(basename $s)
    t=$(date +%s)
    cmake -S $s -B $b -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF \
        2>&1 | grep -E "CPM: Adding package ggml|ggml: |Configuring done|Error" || true
    echo "configure $(basename $s): $(( $(date +%s) - t )) s"
done
ls -d $CPM_SOURCE_CACHE/ggml/patched-*
