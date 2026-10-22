#!/usr/bin/env bash
# Hemoglobin replica s1 again with the C-API build (local helper, not tracked): the MM stages are
# copied from runs/s1 (libtorch build), so only the four ONIOM stages (PET-MAD xs one system,
# PET-OMol s four sites) run, with the metatomic-torch plugin. Compare with runs/s1.
#SBATCH --job-name=hb-capi
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=92G
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/hb-capi-%j.out
set -euo pipefail
M=/home/boittier/metawork
. $M/builds/kuma-env.sh
export GMX=$M/builds/gromacs-capi-kuma-x86_64/bin/gmx
export ONIOM_EXTENSIONS=$M/builds/capi/build-plugin/libmetatomic_torch_plugin.so
HB=$M/gromacs-oniom-hemoglobin
RUN=$HB/runs/s1-capi
echo "== $SLURM_JOB_ID $(hostname) $(date -Is) GMX=$GMX"; nvidia-smi -L
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q H100 || { echo "no H100"; exit 1; }
"$GMX" --version | grep -E 'GROMACS version|Precision|GPU support'
mkdir -p $RUN
for f in $HB/runs/s1/*; do   # MM inputs and outputs only, no ONIOM stage or whole-solute run
    n=$(basename $f)
    [[ $n == o[mn]v[te].* || $n == allml-* || -e $RUN/$n ]] || cp -a $f $RUN/
done
$HB/run.sh $RUN 1
grep -h 'GROMACS version' $RUN/o*.log | sort -u
echo "== done $(date -Is)"
