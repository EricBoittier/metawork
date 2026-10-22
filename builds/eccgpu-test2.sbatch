#!/usr/bin/env bash
# Timing (400 steps) and NVE drift (2 ps) with the GPU embedded Coulomb correction (local helper)
#SBATCH --job-name=eccgpu-test2
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/eccgpu-test2-%j.out
set -uo pipefail
M=/home/boittier/metawork
GMX=$M/builds/gromacs-oniom-kuma-x86_64-eccgpu/bin/gmx
T=$M/builds/eccgpu-test; D=$M/gromacs-oniom-hemoglobin/runs/s1/allml-pet-mad-xs-v1.5.0
for mode in gpu cpu; do
    env $([[ $mode == cpu ]] && echo GMX_METATOMIC_ECC_CPU=1) "$GMX" mdrun -s $D/nvt.tpr -deffnm $T/time400-$mode \
        -nsteps 400 -resetstep 100 -nt 16 -noconfout > /dev/null 2>&1
    echo "== $mode: $(grep -E '^ Metatomic  ' $T/time400-$mode.log | awk '{printf "model %.1f ms", $5/$4*1000}'), $(grep -E '^ Metatomic embedded' $T/time400-$mode.log | awk '{printf "correction %.1f ms", $6/$5*1000}'), $(grep Performance: $T/time400-$mode.log | awk '{print $4" ms/step, "$2" ns/day"}')"
done
"$GMX" mdrun -s $D/nve.tpr -deffnm $T/nve2ps-gpu -nsteps 4000 -nt 16 -noconfout > $T/nve2ps-gpu.out 2>&1
echo "== nve gpu: $(grep Performance: $T/nve2ps-gpu.log | awk '{print $4" ms/step"}')"
