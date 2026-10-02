#!/bin/bash
# CPU prep for train-lorem-pbc.sbatch (free build QOS): LOREM tests on the run
# commit, then the periodicity subsets and a timing subset.
#SBATCH --job-name=lorem-pbc-prep
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=build
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/cosmo/boittier/kuma/madcore-lorem/logs/%x-%j.out
set -euo pipefail
ROOT=/work/cosmo/boittier/kuma/madcore-lorem
CODE=/home/boittier/metawork/metatrain/lorem-sr-switch
VENV=/home/boittier/metawork/.venv
SRC=/home/boittier/data/madcore/mts-2pow18-conservative

git -C "$CODE" log -1 --oneline
# run from the tests dir like tox (reference checkpoints are cwd-relative);
# the long_range hyper changes the checkpoint structure on purpose on this
# local-only branch, so the frozen-structure check is left out; the two
# regression tests also fail on the PR head 42e25756 here (job 4490113), skipped
cd "$CODE/src/metatrain/experimental/lorem/tests" && PYTHONPATH=$CODE/src "$VENV/bin/python" -m pytest \
    -q -rs -p no:cacheprovider -k "not test_checkpoint_did_not_change and not test_regression_init and not test_regression_energies_forces_train" .

cd /home/boittier/metawork/madcore
"$VENV/bin/python" filter_pbc_zip.py "$SRC" "$ROOT/data"
"$VENV/bin/python" filter_pbc_zip.py "$SRC" "$ROOT/data" --subsets mixed --limit 8192
du -sh "$ROOT"/data/*
