#!/usr/bin/env bash
# ============================================================================
# download-datasets.sh
#
# Downloads reference datasets used across the metatensor-ecosystem examples
# into a data directory (default: ~/data) and converts them to extended-XYZ
# (energy in eV, forces in eV/A) ready to hand to metatrain / metatomic /
# ASE examples.
#
# Usage:
#   bash download-datasets.sh [molecule ...]
#   bash download-datasets.sh --data-dir /scratch/data aspirin naphthalene
#
# With no molecules given, downloads just "ethanol". Available MD17
# molecules (classic 8-molecule benchmark, from http://www.sgdml.org):
#   aspirin  benzene2017  ethanol  malonaldehyde
#   naphthalene  salicylic  toluene  uracil
#
# Bigger molecules (MD22 benchmark, same source) also work, e.g.:
#   Ac-Ala3-NHMe  AT-AT  AT-AT-CG-CG  buckyball-catcher  DHA
#   double-walled_nanotube  stachyose
#
# Each molecule downloads as a raw .npz (full trajectory, hundreds of
# thousands of near-duplicate frames -- kept as-is for anyone who wants the
# whole thing) plus a ready-to-use .xyz with a random 1000-frame subsample
# (pass --n-samples to md17_npz_to_xyz.py yourself for a different size).
# Re-running is idempotent -- already-downloaded .npz files are skipped.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPY="$SCRIPT_DIR/../.venv/bin/python"

DATA_DIR="$HOME/data"
molecules=()
while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    *) molecules+=("$1"); shift ;;
  esac
done
[ ${#molecules[@]} -eq 0 ] && molecules=(ethanol)

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

if [ ! -x "$VPY" ]; then
  echo "No venv python at $VPY -- run etc/setup-metawork.sh first (need ase+numpy to convert datasets)." >&2
  exit 1
fi

mkdir -p "$DATA_DIR/md17"
for mol in "${molecules[@]}"; do
  npz="$DATA_DIR/md17/${mol}.npz"
  xyz="$DATA_DIR/md17/${mol}.xyz"

  if [ -f "$npz" ]; then
    log "$mol: already downloaded ($npz)"
  else
    log "Downloading $mol"
    curl -L --fail -o "$npz.tmp" "https://sgdml.org/secure_proxy.php?file=data/npz/md17_${mol}.npz" \
      || curl -L --fail -o "$npz.tmp" "https://sgdml.org/secure_proxy.php?file=repo/datasets/md22_${mol}.npz"
    mv "$npz.tmp" "$npz"
  fi

  log "Converting $mol -> $xyz (1000-frame subsample; energy eV, forces eV/A)"
  "$VPY" "$SCRIPT_DIR/md17_npz_to_xyz.py" "$npz" "$xyz"
done

echo
echo "Datasets in $DATA_DIR/md17/:"
ls -la "$DATA_DIR/md17"
