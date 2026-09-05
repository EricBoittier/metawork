#!/usr/bin/env bash
# Train / eval experimental.lorem on the converted SN2 XYZ.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/data/sn2}"
MTT="${MTT:-$ROOT/metatrain/.tox/lorem-tests/bin/mtt}"
YAML="$ROOT/etc/sn2_zenodo/options/energy-forces-dipole-lorem.yaml"
# Prefer the metatrain checkout (dipole head) over the tox site-packages snapshot.
export PYTHONPATH="$ROOT/metatrain/src${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -x "$MTT" ]; then
  echo "mtt not found at $MTT" >&2
  exit 1
fi
if [ ! -f "$DATA_DIR/sn2.xyz" ]; then
  echo "missing $DATA_DIR/sn2.xyz — run bash etc/sn2_zenodo/convert.sh first" >&2
  exit 1
fi

cd "$DATA_DIR"
RESTART="${RESTART:-}"
if [ -z "$RESTART" ] && [ -f "$DATA_DIR/model.ckpt" ]; then
  RESTART="$DATA_DIR/model.ckpt"
fi
if [ -n "$RESTART" ] && [ "$RESTART" != "0" ]; then
  echo "restarting from $RESTART (lr=$(grep -E 'learning_rate:' "$YAML" | awk '{print $2}'))"
  "$MTT" train "$YAML" --restart "$RESTART"
else
  "$MTT" train "$YAML"
fi
"$MTT" eval model.pt "$ROOT/etc/sn2_zenodo/eval.yaml" -o sn2-eval.xyz
