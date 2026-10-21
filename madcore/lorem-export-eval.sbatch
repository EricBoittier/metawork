#!/bin/bash
# Export the LOREM 100-epoch checkpoints to TorchScript and evaluate them on
# their test sets (the mixed/periodic runs died at export on 10-02, when their
# venv changed under them). The nonperiodic runs, which exported at the time,
# are redone as a check: their test numbers should come back unchanged.
#
#   sbatch ~/metawork/madcore/lorem-export-eval.sbatch [run ...]   (default: all six 100-epoch runs)
#SBATCH --job-name=lorem-export-eval
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --time=02:00:00
#SBATCH --output=/work/cosmo/boittier/kuma/madcore-lorem/logs/%x-%j.out
set -euo pipefail
ROOT=/work/cosmo/boittier/kuma/madcore-lorem
CODE=/home/boittier/metawork/metatrain/runs-e100   # the code the 100-epoch runs trained with
SHA=a73363f3
VENV=/home/boittier/metawork/.venv-lorem   # torch 2.5.1+cu121; make-venv-lorem.sh
export PYTHONPATH=$CODE/src
export LD_LIBRARY_PATH=/home/boittier/data/madcore/lorem-runs/cudart:$VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
[ "$(git -C "$CODE" rev-parse --short=8 HEAD)" = "$SHA" ] || { echo "code is not at $SHA"; exit 1; }
git -C "$CODE" diff --quiet HEAD || { echo "code has uncommitted changes"; exit 1; }
hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), torch.__version__; print(\"torch\", torch.__version__)"
MTT=("$VENV/bin/python" -m metatrain)

RUNS=("$@")
[ ${#RUNS[@]} -gt 0 ] || RUNS=(mixed-lr mixed-sr periodic-lr periodic-sr nonperiodic-lr nonperiodic-sr)
status=0
for r in "${RUNS[@]}"; do
    subset=${r%-*}; lr=$([ "${r##*-}" = lr ] && echo true || echo false)
    run=$ROOT/runs/lorem-$r-100ep; ckpt=$run/lorem-$subset-lr$lr.ckpt; out=$run/eval-1021
    echo "=== $r: $ckpt"
    mkdir -p "$out" && cd "$out"
    if ! "${MTT[@]}" export "$ckpt" -o "lorem-$r-100ep.pt" -e extensions > export.log 2>&1; then
        echo "export FAILED"; tail -20 export.log; status=1; continue
    fi
    cat > eval.yaml <<YAML
systems:
  read_from: $ROOT/data/$subset/test.zip
  length_unit: angstrom
targets:
  energy:
    key: energy
    unit: eV
    forces: true
    stress: false
YAML
    if ! "${MTT[@]}" eval "lorem-$r-100ep.pt" eval.yaml -e extensions -b 16 -o test-predictions.xyz > eval.log 2>&1; then
        echo "eval FAILED"; tail -20 eval.log; status=1; continue
    fi
    grep -iE "rmse|mae" eval.log | tail -6
done
exit $status
