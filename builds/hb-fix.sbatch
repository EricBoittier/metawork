#!/usr/bin/env bash
# Hemoglobin s1's four ONIOM stages on a build with the cut-bond fix (local helper, not tracked):
# BUILD torch (oniom-torch-linkbond) or capi (capi-binding + torch plugin). MM stages copied from
# runs/s1; output in runs/s1-fix-BUILD.   sbatch -A cosmo hb-fix.sbatch BUILD
#SBATCH --job-name=hb-fix
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=92G
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/hb-fix-%j.out
set -euo pipefail
BUILD=$1
M=/home/boittier/metawork
. $M/builds/kuma-env.sh
export GMX=$M/builds/gromacs-$BUILD-linkbond-kuma-x86_64/bin/gmx
[[ $BUILD == capi ]] && export ONIOM_EXTENSIONS=$M/builds/capi/build-plugin/libmetatomic_torch_plugin.so
HB=$M/gromacs-oniom-hemoglobin
RUN=$HB/runs/s1-fix-$BUILD
echo "== $SLURM_JOB_ID $(hostname) $(date -Is) GMX=$GMX"; nvidia-smi -L
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q H100 || { echo "no H100"; exit 1; }
"$GMX" --version | grep -E 'GROMACS version|GPU support'
mkdir -p $RUN
for f in $HB/runs/s1/*; do   # MM inputs and outputs only, no ONIOM stage or whole-solute run
    n=$(basename $f)
    [[ $n == o[mn]v[te].* || $n == allml-* || -e $RUN/$n ]] || cp -a $f $RUN/
done
$HB/run.sh $RUN 1
grep -h 'link caps' $RUN/o*.log | sort -u || true
echo "== done $(date -Is)"
