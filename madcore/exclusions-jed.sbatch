#!/usr/bin/env bash
# make_exclusions.py on jed (CPU): flags -> madcore-data/exclude/ (RESIDUALS = --dump-errors prefix on train)
#SBATCH --job-name=madcore-exclude
#SBATCH --account=cosmo
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail
/home/boittier/metawork/.venv-blackwell/bin/python -I /home/boittier/metawork/madcore/make_exclusions.py \
    /work/cosmo/boittier/lyra/madcore-data/exclude --residuals ${RESIDUALS:?}
