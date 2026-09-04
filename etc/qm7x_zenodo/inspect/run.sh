#!/usr/bin/env bash
# Run inspect/*.py with the metawork venv (so h5py / ase / metatensor resolve).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") SCRIPT [args...]

SCRIPT is list_hdf5, example_read_conf, or summarize_xyz (with or without .py).

Examples:
  bash $QM7X_DIR/inspect/run.sh list_hdf5 ~/data/qm7x/raw/8000.hdf5
  bash $QM7X_DIR/inspect/run.sh example_read_conf ~/data/qm7x/raw/8000.hdf5 --tmap
  bash $QM7X_DIR/inspect/run.sh summarize_xyz ~/data/qm7x/qm7x.xyz
EOF
}

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  usage
  exit 0
fi

script="${1%.py}.py"
shift
py="$QM7X_DIR/inspect/$script"
if [ ! -f "$py" ]; then
  echo "No inspect script $py" >&2
  usage
  exit 1
fi

case "$script" in
  summarize_xyz.py) require_venv ;;
  *) require_h5py ;;
esac
exec "$VPY" "$py" "$@"
