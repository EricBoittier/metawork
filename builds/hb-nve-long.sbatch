#!/usr/bin/env bash
# Long ONIOM NVE of hemoglobin replica s1 (local helper, not tracked), continuing the 5 ps NVE of
# each build: BUILD torch (libtorch, runs/s1) or capi (C API + plugin, runs/s1-capi), or with the
# cut-bond fix torchfix / capifix (runs/s1-fix-torch, runs/s1-fix-capi); MODEL xs
# (PET-MAD xs, sites as one system: onve) or omol (PET-OMol s, four sites: omve); NS nanoseconds.
# One 12 h segment per job: mdrun stops at -maxh and the next segment (afterok) resumes from the
# checkpoint; a segment finding the run complete exits at once.
#   sbatch -A cosmo hb-nve-long.sbatch BUILD MODEL NS
#SBATCH --job-name=hb-nve-long
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=92G
#SBATCH --time=12:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/nve-long-%j.out
set -euo pipefail
BUILD=$1 MODEL=$2 NS=$3
M=/home/boittier/metawork
HB=$M/gromacs-oniom-hemoglobin
. $M/builds/kuma-env.sh
case $BUILD in
    torch) SRC=$HB/runs/s1 ;;
    capi) SRC=$HB/runs/s1-capi GMX=$M/builds/gromacs-capi-kuma-x86_64/bin/gmx
          export ONIOM_EXTENSIONS=$M/builds/capi/build-plugin/libmetatomic_torch_plugin.so ;;
    torchfix) SRC=$HB/runs/s1-fix-torch GMX=$M/builds/gromacs-torch-linkbond-kuma-x86_64/bin/gmx ;;
    capifix) SRC=$HB/runs/s1-fix-capi GMX=$M/builds/gromacs-capi-linkbond-kuma-x86_64/bin/gmx
          export ONIOM_EXTENSIONS=$M/builds/capi/build-plugin/libmetatomic_torch_plugin.so ;;
esac
case $MODEL in xs) STAGE=onve MDP=onve ;; omol) STAGE=omve MDP=onve_omol ;; esac
. $M/etc/oniom/env.sh
L=$HB/runs/s1-long/$BUILD-$MODEL
mkdir -p $L
echo "== $SLURM_JOB_ID $(hostname) $(date -Is) $BUILD $MODEL $NS ns GMX=$GMX"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q H100 || { echo "no H100"; exit 1; }
[[ -f $L/nve.gro ]] && { echo "already complete"; exit 0; }
if [[ ! -f $L/nve.tpr ]]; then
    oniom_mdp $HB/mdp/$MDP.mdp $L/nve.mdp
    sed -i -e "s/^nsteps .*/nsteps                  = $((NS * 2000000))/" \
           -e "s/^nstxout-compressed .*/nstxout-compressed      = 5000/" $L/nve.mdp
    (cd $SRC && "$GMX" grompp -f $L/nve.mdp -c $STAGE.gro -t $STAGE.cpt -p topol.top -n index.ndx \
        -o $L/nve.tpr -po $L/nve.mdout.mdp)
fi
cpi=(); [[ -f $L/nve.cpt ]] && cpi=(-cpi $L/nve.cpt)
"$GMX" mdrun -deffnm $L/nve -nt "$NT" -maxh 11.8 -cpt 30 "${cpi[@]}"
grep -E '^Performance:' $L/nve.log | tail -1
echo "== segment done $(date -Is), complete: $([[ -f $L/nve.gro ]] && echo yes || echo no)"
