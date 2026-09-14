#!/bin/bash -l
#SBATCH --job-name=tiled-vs-mixed
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

RESULTS=$HOME/tiled-vs-mixed
mkdir -p "$RESULTS"

srun --environment=petlr bash -c "
    source \$HOME/petlr/work/tiled_vs_mixed/env.sh
    python bench.py prep --pool ${POOL:-1024} --workers 64 --cache \$SCRATCH/tvm_pool.pkl \
        2>&1 | tee $RESULTS/prep.log
    python sweep.py --cache \$SCRATCH/tvm_pool.pkl --pool ${POOL:-1024} \
        --out $RESULTS/results.json --steps 10 --warmup 3 --start 4 --bisect 4 \
        --compilation-cache \$SCRATCH/tvm_jax_cache \
        2>&1 | tee $RESULTS/sweep.log
"
