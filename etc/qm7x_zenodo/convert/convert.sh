#!/usr/bin/env bash
# Convert one QM7-X shard into the files the training YAMLs expect:
#   $DATA_DIR/qm7x.xyz
#   $DATA_DIR/polarizability_spherical.mts
# (or qm7x.zip with --format zip)
#
# Default is a 200-structure subsample of the small 8000.xz shard.
# Extra arguments are forwarded to zenodo_to_metatensor.py.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra python args]

Download one QM7-X HDF5 shard from Zenodo and convert it to metatrain inputs.

  --data-dir DIR     output directory (default: $DEFAULT_DATA_DIR)
  --cache-dir DIR    xz/hdf5 cache (default: DIR/raw)
  --file NAME        Zenodo filename (default: 8000.xz)
  --n-samples N|all  structures to keep (default: 200)
  --format xyz|zip   (default: xyz)
  --force            rewrite outputs even if they already exist
  -h, --help

Examples:
  bash $QM7X_DIR/convert/convert.sh
  bash $QM7X_DIR/convert/convert.sh --n-samples 500
  bash $QM7X_DIR/convert/convert.sh --n-samples all --format zip
  bash $QM7X_DIR/convert/convert.sh --data-dir /scratch/qm7x --file 1000.xz
EOF
}

DATA_DIR="$DEFAULT_DATA_DIR"
CACHE_DIR=""
FILE="8000.xz"
N_SAMPLES="200"
FORMAT="xyz"
FORCE=0
extra=()

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --file) FILE="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --format) FORMAT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; extra+=("$@"); break ;;
    *) extra+=("$1"); shift ;;
  esac
done

require_h5py
[ -n "$CACHE_DIR" ] || CACHE_DIR="$DATA_DIR/raw"
mkdir -p "$DATA_DIR" "$CACHE_DIR"
fetch_zenodo_info "$CACHE_DIR"

if [ "$FORMAT" = "zip" ]; then
  out="$DATA_DIR/qm7x.zip"
else
  out="$DATA_DIR/qm7x.xyz"
fi

already=0
if [ "$FORMAT" = "zip" ]; then
  [ -e "$out" ] && already=1
elif [ -e "$out" ] && [ -e "$DATA_DIR/polarizability_spherical.mts" ]; then
  already=1
fi
if [ "$already" -eq 1 ] && [ "$FORCE" -eq 0 ]; then
  log "already converted ($out) -- pass --force to rebuild"
  exit 0
fi

log "converting $FILE ($N_SAMPLES samples, format=$FORMAT) -> $DATA_DIR"
"$VPY" "$CONVERT_PY" \
  --file "$FILE" \
  --cache-dir "$CACHE_DIR" \
  --n-samples "$N_SAMPLES" \
  --format "$FORMAT" \
  --output "$out" \
  "${extra[@]+"${extra[@]}"}"

echo
echo "Wrote metatrain inputs in $DATA_DIR"
ls -lh "$DATA_DIR"/qm7x.xyz "$DATA_DIR"/polarizability_spherical.mts "$DATA_DIR"/qm7x.zip 2>/dev/null || true
echo
echo "Train with:"
echo "  bash $QM7X_DIR/train/train.sh --data-dir $DATA_DIR"
