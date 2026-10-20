#!/usr/bin/env bash
# A conservative fine-tune's exported GGUF in mlipcpp-md (FP32): the 2^20 val
# split (compare with ft-baseline 147631: E 93.49, F_direct 717.6, F_cons 2475.1),
# the 2^24 test split with finite differences, and 1 ps conservative NVE on
# crys128 / mol28 (compare with 147623: crys128 drift +-0.02, mol28 2.9 meV/atom).
#   sbatch --export=ALL,FT=/work/cosmo/boittier/lyra/mlipcpp/finetune/cons-ft-train-147625 madcore/ft-check.sbatch
#SBATCH --job-name=ft-check
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail
FT=${FT:?fine-tune directory holding model.gguf}
MD=${MD:-/work/cosmo/boittier/lyra/mlipcpp/build-20261016-115417/bin/mlipcpp-md}
MODEL=$FT/model.gguf
VAL=/home/boittier/data/madcore/memmap-2pow20/val
TEST=/home/boittier/data/madcore/memmap-2pow24/test
OUT=/work/cosmo/boittier/lyra/mlipcpp/runs/ft-check-$SLURM_JOB_ID
mkdir -p $OUT
echo "host $(hostname)  model $MODEL  out $OUT"
echo "=== check: 2^20 val (every 5th of 10452, as ft-baseline 147631)"
$MD check $MODEL $VAL --n 2000 --max-atoms 100000 --fd 0 | grep -E "structures|vs labels"
echo "=== check: 2^24 test, 300 rows + finite differences"
$MD check $MODEL $TEST --n 300 --fd 6 --csv $OUT/check-test.csv | grep -E "structures|vs |finite"
STEPS=${STEPS:-2000}
run() { # tag integrator system-args...
    local tag=$1 integ=$2; shift 2
    echo "=== run $tag: $integ"
    $MD run $MODEL --integrator $integ --steps $STEPS --dt 0.5 --temperature 300 --seed 1 --log-every 50 \
        --thermo $OUT/$tag-$integ.tsv --traj $OUT/$tag-$integ.xyz --traj-every 100 "$@" | \
        awk 'NR<=4 || /graph builds/ || NR%8==0'
}
for integ in conservative respa; do
    run crys128 $integ --data $TEST --index 55621 --repeat 2,2,2
    run mol28 $integ --data $TEST --index 157219
done
echo done
