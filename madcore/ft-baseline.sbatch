#!/usr/bin/env bash
# 145880 (before the conservative fine-tune) on the 2^20 validation split, the
# fine-tune's reference point (mlipcpp-md check, FP32)
#SBATCH --job-name=ft-baseline
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail
MD=/work/cosmo/boittier/lyra/mlipcpp/build-20261016-115417/bin/mlipcpp-md
$MD check /work/cosmo/boittier/lyra/mlipcpp/runs/mlipcpp-step-2pow24-big6adapt16-muon10-145880/model.gguf \
    /home/boittier/data/madcore/memmap-2pow20/val --n 2000 --max-atoms 100000 --fd 0 | grep -E "structures|vs labels"
