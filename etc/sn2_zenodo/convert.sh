#!/usr/bin/env bash
# Download Zenodo 2605341 and write ~/data/sn2/sn2.xyz (10000 structures).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/data/sn2}"
PYTHON="${PYTHON:-$ROOT/metatrain/.tox/lorem-tests/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="${ROOT}/.venv/bin/python"
fi

exec "$PYTHON" "$ROOT/etc/sn2_zenodo/convert.py" --data-dir "$DATA_DIR" "$@"
