#!/usr/bin/env bash
# Restore a PyTorch wheel this machine's NVIDIA driver can actually run,
# then rebuild metatensor-torch / metatomic-torch / featomic-torch against it.
#
# Needed when `uv pip install -e pkg[torch]` has replaced a cu121 (or similar)
# wheel with PyPI's default cu130 build. Safe to re-run.
#
# Usage:
#   bash etc/fix-torch-cuda.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$BASE_DIR/.venv"
VPY="$VENV_DIR/bin/python"

# shellcheck source=etc/_torch_cuda.sh
source "$SCRIPT_DIR/_torch_cuda.sh"

if [ ! -x "$VPY" ]; then
  echo "No venv at $VENV_DIR -- run etc/setup-metawork.sh first." >&2
  exit 1
fi

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

log "Current torch"
"$VPY" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'

log "Installing a driver-matched PyTorch wheel"
torch_before=$("$VPY" -c "import torch; print(torch.__version__)")
ensure_torch_for_driver "$VPY" || true
torch_after=$("$VPY" -c "import torch; print(torch.__version__)")

if [ "$torch_before" != "$torch_after" ]; then
  log "torch changed $torch_before -> $torch_after; rebuilding CUDA extension packages"
  for spec in "metatensor:torch" "metatomic:torch" "featomic:torch"; do
    IFS=':' read -r repo extras <<< "$spec"
    [ -d "$BASE_DIR/$repo" ] || continue
    log "Reinstalling $repo[$extras]"
    uv_pip_keep_torch "$VPY" --reinstall-package "${repo}-torch" \
      -e "$BASE_DIR/$repo[$extras]"
  done
else
  echo "  torch version unchanged"
fi

log "Result"
"$VPY" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 \
   && ! torch_cuda_is_available "$VPY"; then
  echo "CUDA is still unavailable. This driver may be too old for any current wheel," >&2
  echo "or the NVIDIA driver needs updating." >&2
  exit 1
fi
