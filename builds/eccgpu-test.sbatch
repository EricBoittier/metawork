#!/usr/bin/env bash
# GPU vs CPU embedded Coulomb correction on existing inputs (local helper, not tracked)
#SBATCH --job-name=eccgpu-test
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/eccgpu-test-%j.out
set -uo pipefail
M=/home/boittier/metawork
GMX=$M/builds/gromacs-oniom-kuma-x86_64-eccgpu/bin/gmx
T=$M/builds/eccgpu-test; mkdir -p $T
H=$M/gromacs-oniom-hemoglobin/runs/s1
for c in "allml_xs $H/allml-pet-mad-xs-v1.5.0/nvt.tpr" "sites_omol $H/omvt.tpr" \
         "shell_npt $M/builds/smoke/campaign/smoke_box4_shell/npt.tpr"; do
    set -- $c
    for mode in gpu cpu; do
        echo "== $1 $mode"
        env $([[ $mode == gpu ]] && echo GMX_METATOMIC_ECC_CHECK=1 || echo GMX_METATOMIC_ECC_CPU=1) \
            "$GMX" mdrun -s "$2" -deffnm "$T/$1-$mode" -nsteps 20 -nt 16 -noconfout 2>&1 \
            | grep -E 'Embedded Coulomb correction, GPU|Fatal|rror' 
        grep -E '^ Metatomic embedded Cou' "$T/$1-$mode.log"
        grep -A1 'Coulomb \(SR\)' "$T/$1-$mode.log" | head -2 | tail -1
        echo "Pressure-Coulomb: $(grep -A4 '   Energies' "$T/$1-$mode.log" | head -5 | tail -4 | tr -s ' ' | cut -c1-200 | tail -1)"
    done
done
