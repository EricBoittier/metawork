#!/usr/bin/env bash
# ============================================================================
# download-madcore-extxyz.sh
#
# Downloads and decompresses all *.extxyz.gz shards of the MAD-CORE dataset
# from its Materials Cloud Archive record into a data directory
# (default: ~/data/madcore).
#
# Usage:
#   MC_TOKEN=<preview-token> bash download-madcore-extxyz.sh [--data-dir /scratch/data] [--jobs 4] [--keep-gz]
#   bash download-madcore-extxyz.sh --token <preview-token>
#
# The record is currently an unpublished draft, so a preview token is
# required -- pass it via --token or the MC_TOKEN env var (get a fresh link
# with a valid token from the record owner if it has expired; not needed
# once the record is published). The record id can be overridden with
# --record.
#
# Each shard downloads as madcore_<range>.extxyz.gz then is decompressed to
# madcore_<range>.extxyz (the .gz is deleted afterwards; pass --keep-gz to
# keep both). Re-running is idempotent -- shards already extracted are
# skipped, and interrupted downloads resume with curl -C -.
# ============================================================================
set -euo pipefail

RECORD_ID="${MC_RECORD:-91yq0-w3k31}"
TOKEN="${MC_TOKEN:-}"
DATA_DIR="$HOME/data/madcore"
PARALLEL=2
KEEP_GZ=0

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --jobs) PARALLEL="$2"; shift 2 ;;
    --keep-gz) KEEP_GZ=1; shift ;;
    --record) RECORD_ID="$2"; shift 2 ;;
    --token) TOKEN="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

BASE="https://archive.materialscloud.org/api/records/${RECORD_ID}"
mkdir -p "$DATA_DIR"

log "Fetching file list for record ${RECORD_ID}"

fetch_listing() {
  # Try the draft endpoint first (needed for an unpublished/embargoed
  # record), then fall back to the public endpoint for a published one.
  local url code
  if [ -n "$TOKEN" ]; then
    url="${BASE}/draft/files?preview=1&token=${TOKEN}"
    code=$(curl -sS -o "$DATA_DIR/.filelist.json" -w "%{http_code}" "$url" || true)
    if [ "$code" = "200" ]; then
      echo "draft"
      return
    fi
  fi
  url="${BASE}/files"
  code=$(curl -sS -o "$DATA_DIR/.filelist.json" -w "%{http_code}" "$url" || true)
  if [ "$code" = "200" ]; then
    echo "published"
    return
  fi
  echo "none"
}

MODE=$(fetch_listing)
if [ "$MODE" = "none" ]; then
  echo "ERROR: could not fetch file listing (HTTP != 200)." >&2
  echo "If the record is still an unpublished draft, pass a valid --token." >&2
  exit 1
fi
log "Record mode: ${MODE}"

# Extract "key<TAB>content_url<TAB>size" lines for every *.extxyz.gz shard.
python3 - "$DATA_DIR/.filelist.json" > "$DATA_DIR/.extxyz_files.tsv" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
for entry in data.get("entries", []):
    key = entry.get("key", "")
    if key.endswith(".extxyz.gz"):
        url = entry["links"]["content"]
        size = entry.get("size", 0)
        print(f"{key}\t{url}\t{size}")
PY

N=$(wc -l < "$DATA_DIR/.extxyz_files.tsv")
if [ "$N" -eq 0 ]; then
  echo "ERROR: no *.extxyz.gz files found in the record listing." >&2
  exit 1
fi
log "Found ${N} .extxyz.gz shards to download"

download_one() {
  local key="$1" url="$2" size="$3"
  local gz_path="${DATA_DIR}/${key}"
  local xyz_path="${gz_path%.gz}"

  if [ -f "$xyz_path" ]; then
    log "skip ${key} (already extracted -> $(basename "$xyz_path"))"
    return
  fi

  if [ -n "$TOKEN" ] && [[ "$url" == *"/draft/files/"* ]]; then
    url="${url}?preview=1&token=${TOKEN}"
  fi

  log "download ${key}"
  curl -fSL --retry 8 --retry-delay 5 -C - -o "$gz_path" "$url"

  if [ -n "$size" ] && [ "$size" != "0" ]; then
    actual=$(stat -c%s "$gz_path" 2>/dev/null || stat -f%z "$gz_path")
    if [ "$actual" != "$size" ]; then
      echo "ERROR: size mismatch for ${key}: expected ${size}, got ${actual}" >&2
      exit 1
    fi
  fi

  log "extract ${key}"
  gzip -t "$gz_path"
  gzip -dk -f "$gz_path"

  if [ "$KEEP_GZ" -eq 0 ]; then
    rm -f "$gz_path"
  fi
  log "done ${key} -> $(basename "$xyz_path")"
}

running=0
while IFS=$'\t' read -r key url size; do
  download_one "$key" "$url" "$size" &
  running=$((running + 1))
  if [ "$running" -ge "$PARALLEL" ]; then
    wait -n
    running=$((running - 1))
  fi
done < "$DATA_DIR/.extxyz_files.tsv"
wait

rm -f "$DATA_DIR/.filelist.json" "$DATA_DIR/.extxyz_files.tsv"

echo
echo "MAD-CORE extxyz shards in $DATA_DIR:"
ls -la "$DATA_DIR"
