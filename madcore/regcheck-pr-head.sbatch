#!/bin/bash
# Free CPU check: do the LOREM regression tests pass on the unmodified PR head?
#SBATCH --job-name=lorem-regcheck
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=build
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/work/cosmo/boittier/kuma/madcore-lorem/logs/%x-%j.out
set -uo pipefail
VENV=/home/boittier/metawork/.venv
for CODE in /home/boittier/metawork/metatrain/lorem-review /home/boittier/metawork/metatrain/lorem-sr-switch; do
  echo "=== $CODE $(git -C $CODE log -1 --format='%h %s')"
  cd "$CODE/src/metatrain/experimental/lorem/tests" && PYTHONPATH=$CODE/src "$VENV/bin/python" -m pytest \
      -q -p no:cacheprovider -k "regression" . 2>&1 | grep -E "passed|failed|Greatest absolute|^FAILED"
done
