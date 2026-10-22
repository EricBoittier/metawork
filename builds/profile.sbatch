#!/usr/bin/env bash
# Where does each build spend a step? (local helper, not tracked). Hemoglobin s1, 4 heme sites,
# NVE from each fixed build's own 5 ps NVE tpr: PET-MAD xs (one system) and PET-OMol s (four).
# libtorch (gromacs-torch-linkbond) on GPU 0 / cores 0-15 and C API (gromacs-capi-linkbond +
# plugin) on GPU 1 / cores 16-31 at the same time, so both see the same node. Per model: a plain
# run (ns/day), then one with GMX_METATOMIC_TIMER (GROMACS phases) and METATOMIC_TORCH_TIMER
# (plugin phases; synchronizes the GPU, so slower). Free debug QOS, 2 GPUs.
#SBATCH --job-name=hb-profile
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/profile-%j.out
set -uo pipefail
M=/home/boittier/metawork
HB=$M/gromacs-oniom-hemoglobin
. $M/builds/kuma-env.sh
PLUGIN=$M/builds/capi/build-plugin/libmetatomic_torch_plugin.so
STEPS=${STEPS:-4000}
P=$M/builds/profile/$SLURM_JOB_ID
mkdir -p $P && cd $P
echo "== $SLURM_JOB_ID $(hostname) $(date -Is) steps $STEPS"; nvidia-smi -L

one() {  # one BUILD STAGE TAG GPU PINOFFSET [env...]
    local build=$1 stage=$2 tag=$3 gpu=$4 off=$5; shift 5
    local gmx=$M/builds/gromacs-$build-linkbond-kuma-x86_64/bin/gmx d=$P/$build-$stage-$tag
    mkdir -p $d && cd $d
    env "$@" $gmx mdrun -s $HB/runs/s1-fix-$build/$stage.tpr -deffnm run -nsteps $STEPS -ntmpi 1 -ntomp 16 \
        -gpu_id $gpu -pin on -pinoffset $off -pinstride 1 > run.out 2>&1 || echo "   $build $stage $tag FAILED"
}
for stage in onve omve; do
    for tag in plain timer; do
        extra=(); [[ $tag == timer ]] && extra=(GMX_METATOMIC_TIMER=1 METATOMIC_TORCH_TIMER=1)
        one torch $stage $tag 0 0 "${extra[@]}" &
        one capi $stage $tag 1 16 "${extra[@]}" &
        wait
        for build in torch capi; do
            printf '%-6s %-5s %-6s %s\n' $build $stage $tag "$(grep -h '^Performance:' $P/$build-$stage-$tag/run.log | awk '{print $2" ns/day, "$4" ms/step"}')"
        done
    done
done
echo "== GROMACS phases (GMX_METATOMIC_TIMER)"
for f in $P/*-timer/metatomic_timer_rank_0.log; do echo "-- $f"; tail -25 $f; done
echo "== plugin phases (METATOMIC_TORCH_TIMER, C API only)"
for f in $P/capi-*-timer/run.out; do echo "-- $f"; grep -iE 'phase|inputs|forward|backward|outputs' $f | tail -8; done
echo "== done $(date -Is)"
