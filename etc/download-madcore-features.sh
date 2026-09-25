#!/usr/bin/env bash
# ============================================================================
# download-madcore-features.sh
#
# Downloads the madcore_features_*.h5 files (512-dimensional features of
# every MAD-CORE structure, ~29 GB in total) from the Materials Cloud
# Archive record into a data directory (default: ~/data/madcore), and checks
# each against the sha256 listed in the record's index.json.
#
# Usage:
#   MC_TOKEN=<preview-token> bash download-madcore-features.sh [--data-dir DIR] [--jobs 3]
#   bash download-madcore-features.sh --token <preview-token>
#
# Same record and token handling as download-madcore-extxyz.sh (the record is
# an unpublished draft, so a preview token is required until it is
# published). The files are HDF5 with internal compression and are kept as
# they are. Re-running is idempotent: files whose sha256 already matches are
# skipped, and interrupted downloads resume with curl -C -.
# ============================================================================
set -euo pipefail

RECORD_ID="${MC_RECORD:-91yq0-w3k31}"
TOKEN="${MC_TOKEN:-}"
DATA_DIR="$HOME/data/madcore"
PARALLEL=3

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --jobs) PARALLEL="$2"; shift 2 ;;
    --record) RECORD_ID="$2"; shift 2 ;;
    --token) TOKEN="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

BASE="https://archive.materialscloud.org/api/records/${RECORD_ID}"
mkdir -p "$DATA_DIR"
LISTING="$DATA_DIR/.features-filelist.json"
TSV="$DATA_DIR/.features-files.tsv"

fetch_listing() {
  local code
  if [ -n "$TOKEN" ]; then
    code=$(curl -sS -o "$LISTING" -w "%{http_code}" "${BASE}/draft/files?preview=1&token=${TOKEN}" || true)
    if [ "$code" = "200" ]; then echo "draft"; return; fi
  fi
  code=$(curl -sS -o "$LISTING" -w "%{http_code}" "${BASE}/files" || true)
  if [ "$code" = "200" ]; then echo "published"; return; fi
  echo "none"
}

MODE=$(fetch_listing)
if [ "$MODE" = "none" ]; then
  echo "ERROR: could not fetch the file listing." >&2
  echo "If the record is still an unpublished draft, pass a valid --token." >&2
  exit 1
fi
log "Record mode: ${MODE}"

# "key<TAB>content_url<TAB>size<TAB>sha256" for every features file; the
# sha256 comes from index.json next to the data when it is there
python3 - "$LISTING" "$DATA_DIR/index.json" > "$TSV" <<'PY'
import json, os, sys
listing = json.load(open(sys.argv[1]))
index = json.load(open(sys.argv[2]))["files"] if os.path.exists(sys.argv[2]) else {}
for entry in listing.get("entries", []):
    key = entry.get("key", "")
    if key.startswith("madcore_features_") and key.endswith(".h5"):
        sha = index.get(key, {}).get("sha256", "")
        print(f"{key}\t{entry['links']['content']}\t{entry.get('size', 0)}\t{sha}")
PY

N=$(wc -l < "$TSV")
if [ "$N" -eq 0 ]; then
  echo "ERROR: no madcore_features_*.h5 files in the record listing." >&2
  exit 1
fi
log "Found ${N} feature files"

sha_ok() {  # file, expected sha256 (empty: unknown, accept)
  [ -z "$2" ] && return 0
  [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]
}

download_one() {
  local key="$1" url="$2" size="$3" sha="$4"
  local path="${DATA_DIR}/${key}"

  if [ -f "$path" ] && [ "$(stat -c%s "$path")" = "$size" ] && sha_ok "$path" "$sha"; then
    log "skip ${key} (already downloaded, checksum matches)"
    return
  fi

  if [ -n "$TOKEN" ] && [[ "$url" == *"/draft/files/"* ]]; then
    url="${url}?preview=1&token=${TOKEN}"
  fi

  log "download ${key} ($((size / 1000000)) MB)"
  curl -fSL --retry 8 --retry-delay 5 -C - -o "$path" "$url"

  if [ "$size" != "0" ] && [ "$(stat -c%s "$path")" != "$size" ]; then
    echo "ERROR: size mismatch for ${key}: expected ${size}, got $(stat -c%s "$path")" >&2
    exit 1
  fi
  if ! sha_ok "$path" "$sha"; then
    echo "ERROR: sha256 mismatch for ${key}" >&2
    exit 1
  fi
  log "done ${key}"
}

running=0
while IFS=$'\t' read -r key url size sha; do
  download_one "$key" "$url" "$size" "$sha" &
  running=$((running + 1))
  if [ "$running" -ge "$PARALLEL" ]; then
    wait -n
    running=$((running - 1))
  fi
done < "$TSV"
wait

rm -f "$LISTING" "$TSV"
echo
echo "MAD-CORE feature files in $DATA_DIR:"
ls -la "$DATA_DIR"/madcore_features_*.h5
