#!/usr/bin/env bash
# MAD-CORE per-structure labels and checks on jed (CPU): dataset_group /
# dataset_id / ZBL header keys of every row (extract_groups.py -> groups.npz)
# and the geometry scan of the 2^24 memmaps (geometry_scan.py -> geometry/).
# Outputs go to /work. Submit from lyra: ssh jed sbatch <this>.
#SBATCH --job-name=madcore-scan
#SBATCH --account=cosmo
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail

PY=/home/boittier/metawork/.venv-blackwell/bin/python
HERE=/home/boittier/metawork/madcore
DST=/work/cosmo/boittier/lyra/madcore-data
echo "host $(hostname)"
$PY -I $HERE/extract_groups.py /home/boittier/data/madcore -o $DST/groups.npz > $DST/groups.log 2>&1 &
G=$!
$PY -I $HERE/geometry_scan.py $DST/memmap-2pow24 $DST/geometry --workers $((SLURM_CPUS_PER_TASK - 4))
wait $G && tail -5 $DST/groups.log || { echo "!! extract_groups failed"; tail -20 $DST/groups.log; exit 1; }
