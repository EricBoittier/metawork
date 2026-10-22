#!/usr/bin/env bash
# Whole hemoglobin system (51,402 atoms) as ML: model memory and time per model (local helper,
# not tracked). Free debug QOS; submit with -p h100 or -p l40s and matching -c/--mem.
#SBATCH --job-name=hb-fit-all
#SBATCH --qos=debug
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/fit-all-%j.out
set -uo pipefail
M=/home/boittier/metawork
. $M/builds/kuma-env.sh
. $M/etc/oniom/env.sh
echo "== $SLURM_JOB_ID $(hostname) $(date -Is) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
omol='{"charge": 0, "spin_multiplicity": 1}'
for m in ${MODELS:-pet-mad-xs-v1.5.0 pet-mad-s-v1.5.0 pet-omol-s-v1.0.0 pet-omol-m-v1.0.0 pet-mad-m-v1.6.0}; do
    echo "== $m"
    info=(); [[ $m == pet-omol* ]] && info=("$omol")
    "$ONIOM_PY" $M/builds/fit_all.py "$ONIOM_MODELS/$m.pt" "${info[@]}" \
        $M/gromacs-oniom-hemoglobin/runs/s1/npt_whole.gro 2>&1 | grep -v -i warn || echo "   $m failed"
done
echo "== done $(date -Is)"
