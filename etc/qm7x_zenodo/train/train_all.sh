#!/usr/bin/env bash
# Train every compatible options YAML (or a filtered subset).
# Failures are recorded and the rest keep going — experimental.mace /
# dpa3 / space / gap are not in the default metawork venv extras.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run mtt train for each file in $OPTIONS_DIR.

  --data-dir DIR     directory with qm7x.xyz (default: $DEFAULT_DATA_DIR)
  --model NAME       only *-{pet,soap_bpnn,gap,mace,space,dpa3}.yaml
  --endpoint NAME    only {energy,hlgap,dipole,…}-*.yaml (also multi, multi-zip)
  --keep-going       continue after a failed job (default)
  --fail-fast        stop on the first failure
  -h, --help

Examples:
  bash $QM7X_DIR/train/train_all.sh --model pet
  bash $QM7X_DIR/train/train_all.sh --endpoint energy
  bash $QM7X_DIR/train/train_all.sh --model soap_bpnn --endpoint dipole
EOF
}

DATA_DIR="$DEFAULT_DATA_DIR"
MODEL=""
ENDPOINT=""
FAIL_FAST=0

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --keep-going) FAIL_FAST=0; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

require_mtt

if [ ! -e "$DATA_DIR/qm7x.xyz" ] && [ ! -e "$DATA_DIR/qm7x.zip" ]; then
  echo "No qm7x.xyz or qm7x.zip in $DATA_DIR" >&2
  echo "  bash $QM7X_DIR/convert/convert.sh --data-dir $DATA_DIR" >&2
  exit 1
fi

jobs=()
for yaml in "$OPTIONS_DIR"/*.yaml; do
  stem="$(basename "$yaml" .yaml)"
  if [ -n "$MODEL" ]; then
    case "$stem" in
      *-"$MODEL") ;;
      *) continue ;;
    esac
  fi
  if [ -n "$ENDPOINT" ]; then
    case "$stem" in
      "$ENDPOINT"-*) ;;
      *) continue ;;
    esac
  fi
  if [[ "$stem" == *zip* ]] && [ ! -e "$DATA_DIR/qm7x.zip" ]; then
    echo "skip $stem (no qm7x.zip)"
    continue
  fi
  jobs+=("$stem")
done

if [ ${#jobs[@]} -eq 0 ]; then
  echo "No matching options YAMLs." >&2
  exit 1
fi

log "${#jobs[@]} jobs: ${jobs[*]}"

failed=()
passed=()
for stem in "${jobs[@]}"; do
  log "$stem"
  if "$QM7X_DIR/train/train.sh" --data-dir "$DATA_DIR" "$stem"; then
    passed+=("$stem")
  else
    echo "FAILED $stem" >&2
    failed+=("$stem")
    [ "$FAIL_FAST" -eq 1 ] && break
  fi
done

echo
echo "passed (${#passed[@]}): ${passed[*]:-none}"
echo "failed (${#failed[@]}): ${failed[*]:-none}"
[ ${#failed[@]} -eq 0 ]
