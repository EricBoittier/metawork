#!/usr/bin/env bash
# Train / eval experimental.lorem on the converted SN2 XYZ.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/data/sn2}"
MTT="${MTT:-$ROOT/metatrain/.tox/lorem-tests/bin/mtt}"
MTT_PYTHON="${MTT%/*}/python"
YAML="${YAML:-$ROOT/etc/sn2_zenodo/options/energy-forces-dipole-lorem.yaml}"
if [ ! -f "$YAML" ] && [ -f "$ROOT/$YAML" ]; then
  YAML="$ROOT/$YAML"
fi
YAML="$(cd "$(dirname "$YAML")" && pwd)/$(basename "$YAML")"
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

# Return 0 if $1 can be restarted against $YAML; otherwise print mismatches
# and return 1. This checkout's ``mtt train --restart`` keeps the checkpoint
# architecture and ignores yaml model hypers (upstream #1232 is not in
# experimental/lorem yet).
ckpt_matches_yaml() {
  local ckpt="$1"
  "$MTT_PYTHON" - "$ckpt" "$YAML" <<'PY'
import sys

import metatomic.torch  # noqa: F401
import torch
from omegaconf import OmegaConf

ckpt_path, yaml_path = sys.argv[1], sys.argv[2]
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
old = ckpt["model_data"]["model_hypers"]
new = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)[
    "architecture"
]["model"]
skip = {"long_range", "pet"}
mismatches = []
for key, value in new.items():
    if key in skip or key not in old or value is None:
        continue
    if old[key] != value:
        mismatches.append(f"  {key}: ckpt={old[key]!r} yaml={value!r}")
if mismatches:
    print(
        f"checkpoint {ckpt_path} does not match {yaml_path}:",
        file=sys.stderr,
    )
    print("\n".join(mismatches), file=sys.stderr)
    print(
        "this checkout's mtt --restart keeps the checkpoint architecture "
        "and ignores yaml model hypers.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
}

cd "$DATA_DIR"
RESTART="${RESTART:-}"
if [ -z "$RESTART" ] && [ -f "$DATA_DIR/model.ckpt" ]; then
  RESTART="$DATA_DIR/model.ckpt"
fi
if [ -n "$RESTART" ] && [ "$RESTART" != "0" ]; then
  if ! ckpt_matches_yaml "$RESTART"; then
    echo "refusing to restart an incompatible checkpoint." >&2
    echo "to train this yaml from scratch (will overwrite $DATA_DIR/model.ckpt):" >&2
    echo "  mv $DATA_DIR/model.ckpt $DATA_DIR/model-old.ckpt" >&2
    echo "  RESTART=0 bash $ROOT/etc/sn2_zenodo/train.sh" >&2
    exit 1
  fi
  echo "restarting from $RESTART (lr=$(grep -E 'learning_rate:' "$YAML" | awk '{print $2}'))"
  "$MTT" train "$YAML" --restart "$RESTART" "$@"
else
  if [ -f "$DATA_DIR/model.ckpt" ]; then
    echo "training from scratch; will overwrite $DATA_DIR/model.ckpt" >&2
  fi
  "$MTT" train "$YAML" "$@"
fi
EVAL_ARGS=()
if [ -d "$DATA_DIR/extensions" ]; then
  # max_degree > 2 exports sphericart_torch; eval needs that directory.
  EVAL_ARGS+=(-e "$DATA_DIR/extensions")
fi
"$MTT" eval model.pt "$ROOT/etc/sn2_zenodo/eval.yaml" -o sn2-eval.xyz "${EVAL_ARGS[@]}"
