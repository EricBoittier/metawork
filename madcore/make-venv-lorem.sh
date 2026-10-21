#!/usr/bin/env bash
# ~/metawork/.venv-lorem: the LOREM training stack as it was before 10-02 17:11, when
# i-PI 3.3.0 + openmm-ml 1.6 were installed into ~/metawork/.venv and replaced
# torch 2.5.1+cu121 with 2.13.0+cpu (breaking the 100-epoch runs at export).
# A copy of .venv without those, with torch 2.5.1+cu121 back (kuma: CUDA 12.6 driver).
# .venv is a uv venv (no pip): packages go through uv. Safe to rerun: the copy is
# re-synced from .venv first.
set -euo pipefail
SRC=$HOME/metawork/.venv; DST=$HOME/metawork/.venv-lorem
UV=${UV:-$HOME/.local/bin/uv}
rsync -a --delete "$SRC/" "$DST/"
PY=$DST/bin/python
# scripts copied from .venv still point at its interpreter: rewrite the shebangs
grep -lIs "^#!$SRC/bin/python" "$DST"/bin/* | xargs -r sed -i "1s|^#!$SRC/bin/python|#!$PY|"
sed -i "s|$SRC|$DST|g" "$DST"/bin/activate*
"$UV" pip uninstall --python "$PY" ipi openmmml openmm
"$UV" pip install --python "$PY" "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
"$PY" -c "import torch, metatensor.torch, metatomic.torch, sphericart.torch; print(\"torch\", torch.__version__, \"metatomic\", metatomic.torch.__version__)"
"$UV" pip check --python "$PY" || true
echo VENV_DONE
