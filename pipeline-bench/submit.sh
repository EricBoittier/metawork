#!/usr/bin/env bash
# Submit the whole matrix as one GPU job on kuma. Prepares, does not
# submit, unless --submit is passed through to hpc_run.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/etc/hpc_run.py" \
  "$ROOT/etc/hpc-jobs/examples/pipeline-bench.yaml" "$@"
