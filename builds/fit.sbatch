#!/usr/bin/env bash
# GPU memory and time vs ML size on an H100 for each model, with growing parts of hemoglobin
# (gromacs-oniom-hemoglobin/models/fit_test.py; local helper, not tracked)
#SBATCH --job-name=oniom-fit
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/fit-%j.out
set -uo pipefail
M=/home/boittier/metawork
. "$M/etc/oniom/env.sh"
RUN=${RUN:-$M/gromacs-oniom-hemoglobin/runs/smoke}
mkdir -p "$RUN/fit" && cd "$RUN/fit"   # fit_test.py reads ../npt_whole.gro
echo "== $(hostname) $(date -Is) from $RUN"; nvidia-smi -L
omol='{"charge": 0, "spin_multiplicity": 1}'
for m in pet-mad-xs-v1.5.0 pet-mad-s-v1.5.0 pet-mad-m-v1.6.0 pet-omol-s-v1.0.0 pet-omol-m-v1.0.0 pet-omol-l-v1.0.0; do
    echo "== $m"
    info=(); [[ $m == pet-omol* ]] && info=("$omol")
    "$ONIOM_PY" "$M/gromacs-oniom-hemoglobin/models/fit_test.py" "$ONIOM_MODELS/$m.pt" "${info[@]}" 2>&1 \
        | grep -v -i warn || echo "   $m failed"
done
