#!/usr/bin/env bash
# Convert one named QM7-X shard into its own subdirectory:
#   $DATA_DIR/shards/<id>/qm7x.xyz
#   $DATA_DIR/shards/<id>/polarizability_spherical.mts
#
# The training YAMLs still work if you pass that subdirectory as --data-dir.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") SHARD [options] [-- extra python args]

SHARD is 1000, 2000, … 8000 (with or without .xz).

  --data-dir DIR     parent directory (default: $DEFAULT_DATA_DIR)
  --n-samples N|all  structures to keep (default: 200)
  --format xyz|zip   (default: xyz)
  --force            rewrite outputs even if they already exist
  -h, --help

Examples:
  bash $QM7X_DIR/convert/convert_shard.sh 8000
  bash $QM7X_DIR/convert/convert_shard.sh 1000 --n-samples all
  bash $QM7X_DIR/train/train.sh --data-dir $DEFAULT_DATA_DIR/shards/8000
EOF
}

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  usage
  exit 0
fi

SHARD="$1"
shift
SHARD="${SHARD%.xz}"
FILE="${SHARD}.xz"

DATA_DIR="$DEFAULT_DATA_DIR"
N_SAMPLES="200"
FORMAT="xyz"
FORCE=0
extra=()

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --format) FORMAT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; extra+=("$@"); break ;;
    *) extra+=("$1"); shift ;;
  esac
done

ok=0
for s in "${SHARDS[@]}"; do
  [ "$s" = "$SHARD" ] && ok=1
done
if [ "$ok" -eq 0 ]; then
  echo "Unknown shard '$SHARD'. Expected one of: ${SHARDS[*]}" >&2
  exit 1
fi

shard_dir="$DATA_DIR/shards/$SHARD"
mkdir -p "$shard_dir"

convert_args=(
  --data-dir "$shard_dir"
  --cache-dir "$DATA_DIR/raw"
  --file "$FILE"
  --n-samples "$N_SAMPLES"
  --format "$FORMAT"
)
[ "$FORCE" -eq 1 ] && convert_args+=(--force)

exec "$QM7X_DIR/convert/convert.sh" \
  "${convert_args[@]}" \
  "${extra[@]+"${extra[@]}"}"
