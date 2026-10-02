#!/bin/bash
# Free CPU check of the consolidated pipeline branches against upstream/main:
# data/timing utils, PET, and the regression suites the generator change moves.
#SBATCH --job-name=pipeline-tests
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=build
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/cosmo/boittier/kuma/pipeline-consolidated/logs/%x-%j.out
set -uo pipefail
REPO=/home/boittier/metawork/metatrain/pipeline-consolidated
VENV=/home/boittier/metawork/.venv
TMP=/work/cosmo/boittier/kuma/pipeline-consolidated/wt
mkdir -p $TMP
# soap_bpnn/dpa3/mace/space need spex/deepmd/mace/physical_basis, absent from this venv
TESTS="tests/utils/data tests/utils/test_timing.py src/metatrain/pet/tests"
for ref in upstream/main pr/pipeline-timing pr/pipeline-transport perf/memmap-batch-loader-v2 pr/pet-int64-scatter-index; do
  name=${ref//\//_}; wt=$TMP/$name
  git -C $REPO worktree remove --force $wt 2>/dev/null
  git -C $REPO worktree add -q --detach $wt $ref
  cp $REPO/src/metatrain/_version.py $wt/src/metatrain/
  echo "=== $ref $(git -C $wt log -1 --format='%h %s')"
  tests=""; for t in $TESTS; do [ -e "$wt/$t" ] && tests="$tests $t"; done
  (cd $wt && PYTHONPATH=$wt/src timeout 1380 $VENV/bin/python -m pytest -q -p no:cacheprovider -n 0 -W 'ignore:Found metatomic.torch:UserWarning' $tests 2>&1 | grep -E "^(FAILED|ERROR)|^E   [A-Za-z]*Error|passed|failed|error" | tail -25; echo "[exit ${PIPESTATUS[0]}]")
  git -C $REPO worktree remove --force $wt
done
