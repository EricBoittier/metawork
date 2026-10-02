#!/bin/bash
# LOREM training throughput on MAD, real `mtt train`, one GPU, cells run one
# after another so they share the card fairly. Each cell trains EPOCHS epochs
# on SUBSET (default mixed-8192) in its own directory under bench/<tag>/.
#
#   sbatch --export=ALL,CELLS="perf:true:8:0 pipeline:true:8:1" bench-pipeline.sbatch
#   cell = <tree>:<long_range>:<num_workers>:<timing 0|1>[:<-r overrides joined by +>]
#   tree = a worktree name under ~/metawork/metatrain
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=01:30:00
#SBATCH --output=/work/cosmo/boittier/kuma/madcore-lorem/logs/%x-%j.out
set -uo pipefail
: "${CELLS:?}"
SUBSET=${SUBSET:-mixed-8192}
EPOCHS=${EPOCHS:-4}
ROOT=/work/cosmo/boittier/kuma/madcore-lorem
VENV=/home/boittier/metawork/.venv
export LD_LIBRARY_PATH=/home/boittier/data/madcore/lorem-runs/cudart:$VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

for cell in $CELLS; do
    IFS=: read -r tree lr workers timing extra <<< "$cell"
    code=/home/boittier/metawork/metatrain/$tree
    tag=$SLURM_JOB_ID/$tree-lr$lr-w$workers-t$timing${extra:+-$(echo "$extra" | tr -c 'a-zA-Z0-9=.\n' _)}
    run=$ROOT/bench/$tag
    mkdir -p "$run" && cd "$run"
    for split in train val test; do ln -sfn "$ROOT/data/$SUBSET/$split.zip" "$split.zip"; done
    echo "=== CELL $tag code $(git -C "$code" rev-parse --short=8 HEAD) dirty=$(git -C "$code" status --short -uno | wc -l)"
    tenv=(); [ "$timing" = 1 ] && tenv=(METATRAIN_TIMING=1)
    overrides=(); IFS=+ read -ra extras <<< "$extra"; for e in "${extras[@]}"; do overrides+=(-r "$e"); done
    start=$(date +%s.%N)
    env PYTHONPATH=$code/src "${tenv[@]}" \
        "$VENV/bin/mtt" train /home/boittier/metawork/madcore/options-lorem-mts.yaml -o model.pt \
        -r "architecture.model.long_range=$lr" \
        -r "architecture.training.num_epochs=$EPOCHS" \
        -r "architecture.training.num_workers=$workers" \
        "${overrides[@]}" > train.out 2>&1
    echo "=== CELL $tag exit $? wall $(echo "$(date +%s.%N) - $start" | bc) s"
done
