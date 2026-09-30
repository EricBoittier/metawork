#!/usr/bin/env bash
# One GPU per array task (kuma, lyra): each task runs one worker that takes the next free task
# of TASKS. Submit through etc/oniom/submit.sh, which sets the array size and METAWORK.
#SBATCH --job-name=oniom
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
set -uo pipefail
. "$METAWORK/etc/oniom/env.sh"
echo "== $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) ($ONIOM_CLUSTER, ${SLURM_JOB_PARTITION:-?}), GMX $GMX"
nvidia-smi -L || true
bash "$METAWORK/etc/oniom/worker.sh" "$1" 1
