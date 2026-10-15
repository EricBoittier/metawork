#!/usr/bin/env bash
# label_check.py on jed (CPU): energies above a composition baseline vs the
# ZBL repulsion at short contacts -> madcore-data/labelcheck/
#SBATCH --job-name=madcore-labelcheck
#SBATCH --account=cosmo
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail
/home/boittier/metawork/.venv-blackwell/bin/python -I /home/boittier/metawork/madcore/label_check.py /work/cosmo/boittier/lyra/madcore-data/labelcheck
