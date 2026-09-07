#!/usr/bin/env bash
# Train / eval experimental.lorem on the converted SN2 XYZ.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/data/sn2}"
MTT="${MTT:-$(command -v mtt || echo "$ROOT/metatrain/.tox/lorem-tests/bin/mtt")}"
YAML="${YAML:-$ROOT/etc/sn2_zenodo/options/energy-forces-lorem.yaml}"
# eval.yaml requests mtt::dipole too, which only exists on models actually
# trained with a dipole target -- default to the matching eval config based
# on the training YAML's name, override with EVAL_YAML for full control.
if [[ "$YAML" == *dipole* ]]; then
  EVAL_YAML="${EVAL_YAML:-$ROOT/etc/sn2_zenodo/eval.yaml}"
else
  EVAL_YAML="${EVAL_YAML:-$ROOT/etc/sn2_zenodo/eval-energy-forces.yaml}"
fi
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
# Some architectures (e.g. SOAP-BPNN's sphericart_torch) export a TorchScript
# model that needs its extensions/ dir to load; others (LOREM, PET) don't
# produce one at all, and -e on a nonexistent path errors out -- only pass it
# when it's actually there.
EVAL_EXT_ARGS=()
if [ -d "$DATA_DIR/extensions" ]; then
  EVAL_EXT_ARGS=(-e "$DATA_DIR/extensions")
fi
"$MTT" eval model.pt "$EVAL_YAML" "${EVAL_EXT_ARGS[@]}" -o sn2-eval.xyz
