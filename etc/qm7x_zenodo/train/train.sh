#!/usr/bin/env bash
# Run one metatrain job against converted QM7-X inputs.
#
# Working files (qm7x.xyz, polarizability_spherical.mts, qm7x.zip) are
# symlinked into $DATA_DIR/runs/<stem>/ so each job has its own outputs/.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [OPTIONS_YAML]

Train one endpoint × model combination. OPTIONS_YAML can be a stem
(energy-pet), a file name (energy-pet.yaml), or a path. Default: multi-pet.

  --data-dir DIR   directory that contains qm7x.xyz (default: $DEFAULT_DATA_DIR)
  -h, --help

Examples:
  bash $QM7X_DIR/convert/convert.sh
  bash $QM7X_DIR/train/train.sh
  bash $QM7X_DIR/train/train.sh energy-pet
  bash $QM7X_DIR/train/train.sh --data-dir ~/data/qm7x/shards/8000 dipole-soap_bpnn
EOF
}

DATA_DIR="$DEFAULT_DATA_DIR"
YAML_ARG="multi-pet"

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown argument: $1" >&2; usage; exit 1 ;;
    *) YAML_ARG="$1"; shift ;;
  esac
done

resolve_yaml() {
  local arg="$1"
  if [ -f "$arg" ]; then
    printf '%s\n' "$(cd "$(dirname "$arg")" && pwd)/$(basename "$arg")"
    return
  fi
  local name="${arg%.yaml}.yaml"
  if [ -f "$OPTIONS_DIR/$name" ]; then
    printf '%s\n' "$OPTIONS_DIR/$name"
    return
  fi
  echo "No options file for '$arg' (looked at $arg and $OPTIONS_DIR/$name)" >&2
  exit 1
}

require_mtt
YAML="$(resolve_yaml "$YAML_ARG")"
STEM="$(basename "$YAML" .yaml)"
RUN_DIR="$DATA_DIR/runs/$STEM"

if [ ! -e "$DATA_DIR/qm7x.xyz" ] && [ ! -e "$DATA_DIR/qm7x.zip" ]; then
  echo "No qm7x.xyz or qm7x.zip in $DATA_DIR" >&2
  echo "Convert first:" >&2
  echo "  bash $QM7X_DIR/convert/convert.sh --data-dir $DATA_DIR" >&2
  exit 1
fi

if [[ "$STEM" == *zip* ]] && [ ! -e "$DATA_DIR/qm7x.zip" ]; then
  echo "$STEM needs qm7x.zip. Convert with --format zip:" >&2
  echo "  bash $QM7X_DIR/convert/convert.sh --data-dir $DATA_DIR --format zip" >&2
  exit 1
fi

link_train_inputs "$DATA_DIR" "$RUN_DIR"
log "mtt train $YAML  (cwd=$RUN_DIR)"
( cd "$RUN_DIR" && "$MTT" train "$YAML" -o model.pt )

echo
echo "Model:      $RUN_DIR/model.pt"
echo "Checkpoints: $RUN_DIR/outputs/"
