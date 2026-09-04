#!/usr/bin/env bash
# ============================================================================
# setup-metawork.sh
#
# Clones/updates the metatensor-ecosystem repos used in ~/Documents/metawork
# and (re)builds the shared uv-managed Python venv there, installing PyTorch
# appropriate to whatever GPU is actually usable on the machine this runs on.
#
# Safe to re-run: existing repos are `git pull`-ed instead of re-cloned, and
# package installs are no-ops if nothing changed.
# ============================================================================
set -euo pipefail

BASE_DIR="$HOME/Documents/metawork"
VENV_DIR="$BASE_DIR/.venv"
PYTHON_VERSION="3.12"

FORK_OWNER="EricBoittier"     # repos are cloned from your fork when one exists
UPSTREAM_ORG="metatensor"     # falls back to the upstream org otherwise

# Repos that get built + pip-installed (editable) into the venv, in
# dependency order. Format is "name:extras" (extras is a comma list passed
# as .[extras]; leave empty for none).
INSTALL_REPOS=(
  "metatensor:torch"
  "metatomic:torch"
  "featomic:torch"
  "metatrain:soap-bpnn,pet"   # add mace / dpa3 / gap here for more model
                               # types -- they pull in heavier/pinned deps
                               # (e.g. dpa3 needs deepmd-kit, mace pins its
                               # own torch/e3nn versions)
)

# Repos worth having on disk for reference (docs, a header-only helper lib,
# a tiny test model) but not something to pip-install into the venv.
CLONE_ONLY_REPOS=(
  gpu-lite    # header-only CUDA runtime wrapper, used internally by some of
              # the packages above at build time -- nothing to install
  hpc-docs    # metatensor-ecosystem docs for HPC / GPU cluster deployment
  lj-test     # tiny reference metatomic model, useful for smoke-testing
)

# Deliberately NOT cloned by default -- edit the arrays above to add any of
# these back if you need them:
#   *-feedstock repos                          conda-forge packaging
#                                               metadata, nothing to build
#   landing-page, metatensor.github.io,        docs / marketing sites
#     ecosystem-article
#   Workshop-spring-2025                       archived tutorial notebooks
#   metatensor_metatomic_benchmarks            ASV benchmark suite
#   gromacs, lammps, openmm-ml (+ feedstocks)  full external simulation
#                                               codes with their own huge,
#                                               non-Python build systems

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# ---- 1. clone or update a repo, preferring your own fork -------------------
clone_or_update() {
  local repo="$1"
  if [ -d "$BASE_DIR/$repo/.git" ]; then
    log "Updating $repo"
    git -C "$BASE_DIR/$repo" pull --ff-only \
      || echo "  (skipped: local changes or diverged branch -- update $repo by hand)"
    return
  fi
  log "Cloning $repo"
  if git clone "https://github.com/$FORK_OWNER/$repo.git" "$BASE_DIR/$repo" 2>/dev/null; then
    git -C "$BASE_DIR/$repo" remote add upstream "https://github.com/$UPSTREAM_ORG/$repo.git" 2>/dev/null || true
  else
    echo "  (no $FORK_OWNER/$repo fork found -- cloning upstream $UPSTREAM_ORG/$repo instead)"
    git clone "https://github.com/$UPSTREAM_ORG/$repo.git" "$BASE_DIR/$repo"
  fi
}

mkdir -p "$BASE_DIR"
for entry in "${INSTALL_REPOS[@]}"; do
  clone_or_update "${entry%%:*}"
done
for repo in "${CLONE_ONLY_REPOS[@]}"; do
  clone_or_update "$repo"
done

# ---- 1b. known upstream build-bug patches -----------------------------------
# featomic_torch's setup.py assumes `nvidia.cudnn.__file__` is always a real
# path, but on some CUDA-13 wheel builds `nvidia.cudnn` is a namespace
# package (no __init__.py) and __file__ is None, which turns the intended
# ImportError fallback into an unhandled TypeError at build time. Patch it
# to fall through to the existing fallback in that case. No-op if the repo
# isn't installed here, if upstream has already fixed it, or if we already
# patched it on a previous run.
patch_featomic_cudnn_namespace_pkg_bug() {
  local f="$BASE_DIR/featomic/python/featomic_torch/setup.py"
  [ -f "$f" ] || return 0
  python3 - "$f" <<'PYEOF'
import sys

path = sys.argv[1]
old = "            cudnn_root = os.path.dirname(nvidia.cudnn.__file__)\n"
new = (
    "            # FEATOMIC_CUDNN_NAMESPACE_PKG_FIX: nvidia.cudnn can be a namespace\n"
    "            # package (no __init__.py) on some CUDA-13 wheel builds, in which\n"
    "            # case __file__ is None and os.path.dirname() raises TypeError\n"
    "            # instead of the ImportError this try/except expects.\n"
    "            if nvidia.cudnn.__file__ is None:\n"
    "                raise ImportError(\"nvidia.cudnn has no __file__ (namespace package)\")\n"
    "            cudnn_root = os.path.dirname(nvidia.cudnn.__file__)\n"
)

text = open(path).read()
if old in text:
    open(path, "w").write(text.replace(old, new, 1))
    print("  patched", path)
PYEOF
}
patch_featomic_cudnn_namespace_pkg_bug

# ---- 2. build toolchain sanity check ---------------------------------------
log "Checking build toolchain (these repos compile Rust/C++ extensions)"
missing=0
for tool in rustc cargo cmake gcc; do
  if ! command -v "$tool" >/dev/null; then
    echo "  MISSING: $tool"
    missing=1
  fi
done
[ "$missing" = 0 ] && echo "  rustc, cargo, cmake, gcc all present"

# `cargo` existing on PATH isn't enough: on some machines it's a rustup shim
# with no default toolchain configured, which makes it print an error
# instead of a version -- and that in turn breaks metatensor-core's CMake
# version check in a confusing way. Detect and self-heal that specific case.
if [ "$missing" = 0 ] && ! cargo --version >/dev/null 2>&1; then
  if command -v rustup >/dev/null; then
    echo "  cargo is a rustup shim with no default toolchain -- running 'rustup default stable'"
    rustup default stable
  else
    echo "  cargo does not run and rustup isn't available to fix it -- builds below will fail" >&2
  fi
fi

# ---- 3. venv ----------------------------------------------------------------
if ! command -v uv >/dev/null; then
  log "Installing uv (https://astral.sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null; then
    echo "uv install ran but 'uv' still isn't on PATH -- open a new shell and re-run this script." >&2
    exit 1
  fi
fi
if [ ! -d "$VENV_DIR" ]; then
  log "Creating venv at $VENV_DIR"
  uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi
VPY="$VENV_DIR/bin/python"

# ---- 4. GPU-aware PyTorch install -------------------------------------------
log "Detecting GPU / CUDA driver"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "  Working NVIDIA driver detected:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/    /'
  echo "  Installing default (CUDA-enabled) PyTorch wheel"
  uv pip install --python "$VPY" "torch>=2.7"
else
  echo "  No working NVIDIA driver found (this is the case on nouveau-only"
  echo "  boxes like this one) -- installing the smaller CPU-only wheel."
  echo "  Re-run this script unmodified on a machine with a real NVIDIA"
  echo "  driver to get CUDA-accelerated PyTorch instead."
  uv pip install --python "$VPY" "torch>=2.7" --index-url https://download.pytorch.org/whl/cpu
fi

# ---- 4b. CUDA toolkit (nvcc) detection --------------------------------------
# metatensor-torch / metatomic-torch / featomic compile actual CUDA kernels
# at build time, which needs `nvcc`, not just a driver. On machines where
# CUDA is installed but not wired into PATH by default (common when it's not
# loaded via an environment module), find it and export it here.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 && ! command -v nvcc >/dev/null 2>&1; then
  log "nvidia-smi works but nvcc isn't on PATH -- looking for a CUDA toolkit install"
  for cuda_dir in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-*; do
    if [ -n "$cuda_dir" ] && [ -x "$cuda_dir/bin/nvcc" ]; then
      echo "  Found CUDA toolkit at $cuda_dir -- exporting PATH/CUDA_HOME/CUDACXX"
      export CUDA_HOME="$cuda_dir"
      export CUDACXX="$cuda_dir/bin/nvcc"
      export PATH="$cuda_dir/bin:$PATH"
      export LD_LIBRARY_PATH="$cuda_dir/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      break
    fi
  done
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "  No CUDA toolkit found in the usual locations -- CUDA kernel builds below may fail." >&2
    echo "  If it's installed somewhere nonstandard, export CUDA_HOME before running this script." >&2
  fi
fi

# ---- 5. install the ecosystem packages, in dependency order ----------------
for entry in "${INSTALL_REPOS[@]}"; do
  repo="${entry%%:*}"
  extras="${entry#*:}"
  target="$BASE_DIR/$repo"
  [ -n "$extras" ] && target="$target[$extras]"
  log "Installing $repo${extras:+ [$extras]}"
  uv pip install --python "$VPY" -e "$target"
done

# ---- 6. summary --------------------------------------------------------------
log "Summary"
"$VPY" - <<'EOF'
import importlib
import torch

print(f"torch {torch.__version__}  (cuda build: {torch.version.cuda}, "
      f"cuda available at runtime: {torch.cuda.is_available()})")

for mod in ("metatensor", "metatensor.torch", "metatomic", "metatomic.torch", "featomic", "metatrain"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:20s} ok  ({getattr(m, '__version__', '')})")
    except Exception as e:
        print(f"{mod:20s} FAILED: {e}")
EOF

echo
echo "Activate with:  source $VENV_DIR/bin/activate"
echo "Or run one-off: uv run --python $VPY <command>"
