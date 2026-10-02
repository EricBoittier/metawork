#!/bin/bash
# experimental.lorem on MAD-CORE level 18, by periodicity and with/without the
# long-range block. Code: local-only branch local/lorem-long-range-switch
# (PR #1265 + a long_range switch; never pushed). Data: filter_pbc_zip.py.
#
#   sbatch --export=ALL,SUBSET=mixed,LONG_RANGE=true  -J lorem-mixed-lr  train-lorem-pbc.sbatch
#   SUBSET      mixed | periodic | nonperiodic (or a -N timing subset)
#   LONG_RANGE  true | false
#   EPOCHS      default 10 (options-lorem-mts.yaml)
#
# The run directory is named after the job name, not the job id, so a
# resubmission with the same -J continues from the last epoch checkpoint.
# Segments (dispatch_lorem_pbc.py) chain with afterany and PREV_JOB set: a
# segment exits at once when the run is done or the previous one crashed.
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --output=/work/cosmo/boittier/kuma/madcore-lorem/logs/%x-%j.out
set -euo pipefail
: "${SUBSET:?}" "${LONG_RANGE:?}"
EPOCHS=${EPOCHS:-10}

ROOT=/work/cosmo/boittier/kuma/madcore-lorem
CODE=${CODE:-/home/boittier/metawork/metatrain/lorem-sr-switch}
SHA=${SHA:-47c25cc3}   # 47c25cc3 = 060f69c3 + Bernstein cutoff NaN fix (0a2d45d4 on the PR)
VENV=/home/boittier/metawork/.venv   # torch 2.5.1+cu121: kuma's driver is CUDA 12.6
export PYTHONPATH=$CODE/src
# sphericart/nvrtc dlopen unversioned names; the venv only ships .so.12
export LD_LIBRARY_PATH=/home/boittier/data/madcore/lorem-runs/cudart:$VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# never run on code that moved under us
[ "$(git -C "$CODE" rev-parse --short=8 HEAD)" = "$SHA" ] || { echo "code is not at $SHA"; exit 1; }
git -C "$CODE" diff --quiet HEAD || { echo "code has uncommitted changes"; exit 1; }

RUN=$ROOT/runs/$SLURM_JOB_NAME
MODEL=lorem-$SUBSET-lr$LONG_RANGE.pt
mkdir -p "$RUN" && cd "$RUN"
[ -e "$MODEL" ] && { echo "run finished earlier ($MODEL), nothing to do"; exit 0; }
if [ -n "${PREV_JOB:-}" ]; then
    prev=$(sacct -j "$PREV_JOB" -X -n -o state | tr -d ' ')
    case "$prev" in TIMEOUT|COMPLETED) ;; *) echo "previous segment $PREV_JOB ended $prev, stopping"; exit 1;; esac
fi
for split in train val test; do ln -sfn "$ROOT/data/$SUBSET/$split.zip" "$split.zip"; done

hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), torch.__version__"
echo "code $CODE @ $SHA subset=$SUBSET long_range=$LONG_RANGE epochs=$EPOCHS"

restart=()
compgen -G "outputs/*/*/model_*.ckpt" > /dev/null && restart=(--restart auto)
# composition baseline and scaler depend only on the training set: reuse the
# ones fitted once per subset ($ROOT/cache/<subset>, copied from a finished
# run) instead of refitting them over the whole set in every run and segment
cached=()
if [ -f "$ROOT/cache/$SUBSET/composition_model.ckpt" ] && [ -f "$ROOT/cache/$SUBSET/scaler.ckpt" ]; then
    cached=(-r "architecture.training.atomic_baseline=$ROOT/cache/$SUBSET/composition_model.ckpt"
            -r "architecture.training.fixed_scaling_weights=$ROOT/cache/$SUBSET/scaler.ckpt")
    echo "using cached composition/scaler from $ROOT/cache/$SUBSET"
fi
start=$(date +%s)
"$VENV/bin/mtt" train /home/boittier/metawork/madcore/options-lorem-mts.yaml \
    -o "$MODEL" -e extensions "${restart[@]}" "${cached[@]}" \
    -r "architecture.model.long_range=$LONG_RANGE" \
    -r "architecture.training.num_epochs=$EPOCHS" "$@"
echo "ELAPSED $(( $(date +%s) - start ))"
# done: drop the segments still waiting behind this one
scancel -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING || true
