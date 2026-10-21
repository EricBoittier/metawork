#!/usr/bin/env bash
# Compare the plugin on a GGUF file with the TorchScript export of the
# checkpoint it came from, on a few periodic, slab and molecular structures.
#
#   compare-torch.sh <model.gguf> <model.pt|model.ckpt>
#
# TORCH_PYTHON must have metatomic-torch (and metatrain to export a .ckpt);
# metatomic-core and metatomic-torch both import as `metatomic`, hence two venvs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/build"
TORCH_PYTHON=${TORCH_PYTHON:-$HERE/../../.venv-upet/bin/python}
GGUF=$1
MODEL=$2
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

if [[ "$MODEL" == *.ckpt ]]; then
  (cd "$WORK" && "$(dirname "$TORCH_PYTHON")/mtt" export "$MODEL" -o model.pt > export.log 2>&1)
  MODEL="$WORK/model.pt"
fi

"$BUILD/venv/bin/python" "$HERE/structures.py" "$WORK"
XYZ=("$WORK"/{si,slab,nacl,fe,cluster,water}.xyz)
"$BUILD/venv/bin/python" "$HERE/plugin_eval.py" "$BUILD/mlipcpp-metatomic.so" "$GGUF" "$WORK/plugin.npz" "${XYZ[@]}" > /dev/null
"$TORCH_PYTHON" -W ignore "$HERE/torch_eval.py" "$MODEL" "$WORK/plugin.npz" "${XYZ[@]}" 2>&1 | grep -v "^Warp"
