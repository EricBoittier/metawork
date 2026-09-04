#!/usr/bin/env bash
# Convert every QM7-X Zenodo shard (1000.xz … 8000.xz).
#
# Default keeps 200 random structures per shard so a first run is cheap.
# Pass --n-samples all for the full ~4.2 M structures — that downloads
# ~9.6 GB compressed and needs tens of GB uncompressed. See README.md.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Convert all eight QM7-X shards into \$DATA_DIR/shards/<id>/.

  --data-dir DIR     parent directory (default: $DEFAULT_DATA_DIR)
  --n-samples N|all  structures per shard (default: 200)
  --format xyz|zip   (default: xyz)
  --force            rebuild shards that already have outputs
  -h, --help

Compressed sizes (Zenodo record 4288677):

  1000.xz  ~682 MB     5000.xz  ~1.1 GB
  2000.xz  ~995 MB     6000.xz  ~1.9 GB
  3000.xz  ~1.9 GB     7000.xz  ~1.0 GB
  4000.xz  ~1.4 GB     8000.xz  ~85 MB    (recommended first shard)

Examples:
  bash $QM7X_DIR/convert/convert_all_shards.sh
  bash $QM7X_DIR/convert/convert_all_shards.sh --n-samples 500
  bash $QM7X_DIR/convert/convert_all_shards.sh --n-samples all   # full dump
EOF
}

DATA_DIR="$DEFAULT_DATA_DIR"
N_SAMPLES="200"
FORMAT="xyz"
FORCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --format) FORMAT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [ "$N_SAMPLES" = "all" ]; then
  log "FULL conversion: ~9.6 GB download + large HDF5/XYZ on disk"
fi

failed=()
for shard in "${SHARDS[@]}"; do
  log "shard $shard"
  args=( "$QM7X_DIR/convert/convert_shard.sh" "$shard"
         --data-dir "$DATA_DIR"
         --n-samples "$N_SAMPLES"
         --format "$FORMAT" )
  [ "$FORCE" -eq 1 ] && args+=(--force)
  if ! "${args[@]}"; then
    echo "  FAILED: shard $shard" >&2
    failed+=("$shard")
  fi
done

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "All shards converted under $DATA_DIR/shards/"
  ls -lh "$DATA_DIR/shards"/*/qm7x.xyz "$DATA_DIR/shards"/*/qm7x.zip 2>/dev/null || true
else
  echo "Failed shards: ${failed[*]}" >&2
  exit 1
fi
