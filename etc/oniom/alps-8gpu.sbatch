#!/usr/bin/env bash
# Eight GPUs in one allocation (daint, clariden: 2 GH200 nodes x 4): eight workers, one per
# GPU, share TASKS and keep taking tasks until none are left or time runs out. Submit
# through etc/oniom/submit.sh, which adds the account and the uenv.
#SBATCH --job-name=oniom
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
set -uo pipefail
# with non-bondeds and PME on the GPU, more than ~32 OpenMP threads per run doesn't pay off
export NT=${NT:-32}
. "$METAWORK/etc/oniom/env.sh"
export CUDA_CACHE_DISABLE=1   # no JIT cache on the shared file system
echo "== $SLURM_JOB_ID on $SLURM_JOB_NODELIST ($ONIOM_CLUSTER), $SLURM_NTASKS workers, GMX $GMX"
for i in $(seq 1 "$SLURM_NTASKS"); do
    srun --ntasks=1 --nodes=1 --exact --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --gpu-bind=single:1 bash "$METAWORK/etc/oniom/worker.sh" "$1" &
done
wait
