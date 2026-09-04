# Shared PyTorch CUDA-channel helpers. Sourced by setup-metawork.sh and
# fix-torch-cuda.sh. Callers set VPY (and optionally VENV_DIR / TORCH_PIN_FILE)
# before calling the install functions.
#
# Later `uv pip install -e pkg[torch]` will happily replace a cu121 wheel with
# PyPI's default cu130 build: the versions both satisfy `torch`, and uv does
# not care which CUDA runtime is inside. Pin the installed torch and pass
# --extra-index-url so that does not happen.

# Newest first. Empty string = PyPI default (currently a CUDA 13 wheel).
TORCH_CUDA_CHANNELS=(
  ""
  "https://download.pytorch.org/whl/cu126"
  "https://download.pytorch.org/whl/cu124"
  "https://download.pytorch.org/whl/cu121"
  "https://download.pytorch.org/whl/cu118"
)

TORCH_INDEX_URL="${TORCH_INDEX_URL-}"

driver_max_cuda() {
  nvidia-smi 2>/dev/null \
    | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' \
    | grep -oE '[0-9]+\.[0-9]+' \
    | head -1
}

index_url_for_driver_cuda() {
  local v="$1"
  awk -v v="$v" 'BEGIN {
    v += 0
    if (v >= 12.8)      print ""
    else if (v >= 12.6) print "https://download.pytorch.org/whl/cu126"
    else if (v >= 12.4) print "https://download.pytorch.org/whl/cu124"
    else if (v >= 12.1) print "https://download.pytorch.org/whl/cu121"
    else                print "https://download.pytorch.org/whl/cu118"
  }'
}

torch_cuda_is_available() {
  local py="${1:-$VPY}"
  [ "$("$py" -c 'import torch; print("True" if torch.cuda.is_available() else "False")' 2>/dev/null | tail -1)" = "True" ]
}

install_torch_from_index() {
  local py="$1"
  local url="${2-}"
  if [ -n "$url" ]; then
    echo "  uv pip install torch --index-url $url"
    uv pip install --python "$py" --reinstall-package torch torch --index-url "$url"
  else
    echo "  uv pip install torch (PyPI default CUDA wheel)"
    uv pip install --python "$py" --reinstall-package torch torch
  fi
}

pin_torch() {
  local py="$1"
  local dest="${TORCH_PIN_FILE:-${VENV_DIR:-.}/.torch-constraint.txt}"
  mkdir -p "$(dirname "$dest")"
  "$py" -c "import torch; print('torch==' + torch.__version__)" > "$dest"
  echo "  pinned $("$py" -c 'import torch; print(torch.__version__)') -> $dest"
}

# Install a torch build this driver can initialize. Starts at the channel
# matching nvidia-smi's CUDA Version, then walks older channels if
# torch.cuda.is_available() is still False (nvidia-smi header can be ahead
# of what the running driver actually supports).
ensure_torch_for_driver() {
  local py="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "  No working NVIDIA driver -- installing CPU torch"
    uv pip install --python "$py" --reinstall-package torch torch \
      --index-url https://download.pytorch.org/whl/cpu
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    pin_torch "$py"
    return 0
  fi

  echo "  Working NVIDIA driver detected:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/    /'

  local ver mapped
  ver="$(driver_max_cuda)"
  mapped="$(index_url_for_driver_cuda "${ver:-0}")"
  echo "  Driver supports up to CUDA ${ver:-unknown} -> ${mapped:-PyPI default}"

  local -a walk=()
  local seen=0 ch
  for ch in "${TORCH_CUDA_CHANNELS[@]}"; do
    if [ "$seen" -eq 0 ]; then
      if [ "$ch" = "$mapped" ]; then
        seen=1
        walk+=("$ch")
      fi
    else
      walk+=("$ch")
    fi
  done
  if [ "$seen" -eq 0 ]; then
    walk=("${TORCH_CUDA_CHANNELS[@]}")
  fi

  for ch in "${walk[@]}"; do
    install_torch_from_index "$py" "$ch"
    if torch_cuda_is_available "$py"; then
      TORCH_INDEX_URL="$ch"
      pin_torch "$py"
      echo "  torch.cuda.is_available() = True ($("$py" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)'))"
      return 0
    fi
    echo "  torch.cuda.is_available() = False with this wheel -- trying an older CUDA channel"
  done

  echo "  WARNING: no PyTorch wheel initialized CUDA on this driver." >&2
  TORCH_INDEX_URL="$mapped"
  pin_torch "$py"
  return 1
}

# Install packages without replacing the driver-matched torch wheel.
uv_pip_keep_torch() {
  local py="$1"
  shift
  local args=()
  if [ -n "${TORCH_INDEX_URL:-}" ]; then
    args+=(--extra-index-url "$TORCH_INDEX_URL")
    # uv's default index-strategy ("first-index") stops looking for a
    # package as soon as *any* configured index has it, even if that index
    # only carries old releases. download.pytorch.org/whl/* mirrors plain
    # PyPI packages too (e.g. packaging, up to whatever version torch's own
    # build needed at the time), which silently caps things like
    # `packaging` well below what other packages we build here need at
    # build time (e.g. setuptools >=77 requires packaging >=24.2, and some
    # setup.py scripts use packaging APIs added in 26.0), causing
    # hard-to-diagnose build isolation failures. unsafe-best-match makes uv
    # consider all indexes and pick the best version instead.
    args+=(--index-strategy unsafe-best-match)
  fi
  local pin="${TORCH_PIN_FILE:-${VENV_DIR:-.}/.torch-constraint.txt}"
  if [ -f "$pin" ]; then
    args+=(-c "$pin")
  fi
  uv pip install --python "$py" "${args[@]}" "$@"
}
