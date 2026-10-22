#!/usr/bin/env bash
# Smoke test of the kuma builds: every stage capped at CAP steps (local helper, not tracked)
#SBATCH --job-name=oniom-smoke
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/smoke-%j.out
set -euo pipefail
M=/home/boittier/metawork
. "$M/etc/oniom/env.sh"
export CAP=${CAP:-200} GMX_REAL=$GMX
echo "== $(hostname) $(date -Is) GMX=$GMX GMX_D=$GMX_D NT=$NT"; nvidia-smi -L
"$GMX" --version | grep -E 'Precision|GPU support|SIMD'
GMX=$M/builds/gmx-capped "$M/gromacs-oniom-hemoglobin/run.sh" "$M/gromacs-oniom-hemoglobin/runs/smoke" 1
echo "== hemoglobin smoke done $(date -Is)"
export JOBS=$M/builds/smoke/campaign
GMX=$M/builds/gmx-capped "$M/gromacs-oniom-cyclic/campaign/run_job.sh" smoke_box4_shell 4.0 shell 1
GMX_REAL=$GMX_D GMX=$M/builds/gmx-capped "$M/gromacs-oniom-cyclic/campaign/run_job.sh" smoke_box4_shellres 4.0 shellres 1
echo "== campaign smoke done $(date -Is)"
