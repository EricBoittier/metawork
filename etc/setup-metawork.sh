#!/usr/bin/env bash
# ============================================================================
# setup-metawork.sh
#
# Clones/updates the metatensor-ecosystem repos and (re)builds the shared
# uv-managed Python venv, both rooted at the parent of this script's own
# etc/ directory (so this works wherever the project folder actually lives
# -- ~/metawork, ~/Documents/metawork, wherever you put it -- as long as
# this script stays inside its etc/ subdirectory) -- installing PyTorch
# appropriate to whatever GPU is actually usable on the machine this runs on.
#
# Safe to re-run: existing repos are `git pull`-ed instead of re-cloned, and
# package installs are no-ops if nothing changed.
# ============================================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$BASE_DIR/.venv"
PYTHON_VERSION="3.12"

FORK_OWNER="EricBoittier"     # repos are cloned from your fork when one exists
UPSTREAM_ORG="metatensor"     # falls back to the upstream org otherwise

# Repos that get built + pip-installed (editable) into the venv, in
# dependency order. Format is "name:extras:org" -- extras is a comma list
# passed as .[extras] (empty for none), org is the upstream GitHub org/user
# this repo lives under (empty defaults to $UPSTREAM_ORG).
INSTALL_REPOS=(
  "metatensor:torch:"
  "metatomic:torch:"   # NOT ",torchsim" -- metatomic-torchsim currently
                        # pins vesin<0.6 while metatomic-ase (pulled in by
                        # the torch extra) pins vesin>=0.6, so uv can't
                        # resolve both together. Upstream version-pin bug,
                        # see Known Issues in the README. Try re-adding
                        # torchsim here once that's fixed upstream.
  "featomic:torch:"
  "metatrain:soap-bpnn,pet:"   # add mace / dpa3 / gap here for more model
                               # types -- they pull in heavier/pinned deps
                               # (e.g. dpa3 needs deepmd-kit, mace pins its
                               # own torch/e3nn versions)
  "i-pi::i-pi"                 # pure-Python force engine, no special build
  "chemiscope::lab-cosmo"       # structure/property viewer widget; its
                                 # build runs `npm` to bundle JS assets, see
                                 # the toolchain check below
)

# Repos worth having on disk for reference (docs, a header-only helper lib,
# a tiny test model, or an engine with its own separate native build system)
# but not something to pip-install into the venv. Format is "name:org" (org
# empty defaults to $UPSTREAM_ORG).
CLONE_ONLY_REPOS=(
  "gpu-lite:"    # header-only CUDA runtime wrapper, used internally by some
                 # of the packages above at build time -- nothing to install
  "hpc-docs:"    # metatensor-ecosystem docs for HPC / GPU cluster deployment
  "lj-test:"     # tiny reference metatomic model, useful for smoke-testing
  "lammps:"      # metatomic-enabled LAMMPS fork -- build per
                 # https://docs.metatensor.org/metatomic/latest/engines/lammps.html
  "gromacs:"     # metatomic staging branch for GROMACS -- build per
                 # https://docs.metatensor.org/metatomic/latest/engines/gromacs.html
  "eOn:TheochemUI"    # transition-state/eOn engine, metatomic support is in
                      # the official version -- build per
                      # https://docs.metatensor.org/metatomic/latest/engines/eon.html
  "plumed2:plumed"    # PLUMED, metatomic support is in the official
                      # (development) version -- build per
                      # https://docs.metatensor.org/metatomic/latest/engines/plumed.html
)

# Deliberately NOT cloned by default -- edit the arrays above to add any of
# these back if you need them:
#   *-feedstock repos                          conda-forge packaging
#                                               metadata, nothing to build
#   landing-page, metatensor.github.io,        docs / marketing sites
#     ecosystem-article
#   Workshop-spring-2025                       archived tutorial notebooks
#   metatensor_metatomic_benchmarks            ASV benchmark suite
#   openmm-ml (+ feedstock)                    another full external
#                                               simulation code, same
#                                               reasoning as lammps/gromacs

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# ---- 1. clone or update a repo, preferring your own fork -------------------
# org defaults to $UPSTREAM_ORG when empty (i.e. "" or omitted).
clone_or_update() {
  local repo="$1"
  local org="${2:-$UPSTREAM_ORG}"
  [ -z "$org" ] && org="$UPSTREAM_ORG"
  if [ -d "$BASE_DIR/$repo/.git" ]; then
    log "Updating $repo"
    git -C "$BASE_DIR/$repo" pull --ff-only \
      || echo "  (skipped: local changes or diverged branch -- update $repo by hand)"
    return
  fi
  log "Cloning $repo"
  # An empty GitHub fork still clones successfully, but has no HEAD -- git
  # add of the parent repo then fails with "does not have a commit checked
  # out". Treat that the same as "no fork".
  if git clone "https://github.com/$FORK_OWNER/$repo.git" "$BASE_DIR/$repo" 2>/dev/null \
     && git -C "$BASE_DIR/$repo" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "$BASE_DIR/$repo" remote add upstream "https://github.com/$org/$repo.git" 2>/dev/null || true
  else
    rm -rf "$BASE_DIR/$repo"
    echo "  (no usable $FORK_OWNER/$repo fork -- cloning upstream $org/$repo instead)"
    git clone "https://github.com/$org/$repo.git" "$BASE_DIR/$repo"
  fi
}

mkdir -p "$BASE_DIR"
for entry in "${INSTALL_REPOS[@]}"; do
  IFS=':' read -r repo _extras org <<< "$entry"
  clone_or_update "$repo" "$org"
done
for entry in "${CLONE_ONLY_REPOS[@]}"; do
  IFS=':' read -r repo org <<< "$entry"
  clone_or_update "$repo" "$org"
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

# npm/node are only needed for chemiscope (it bundles its JS widget assets
# via npm at build time, and requires node >=20) -- not a hard requirement
# for everything else, so just record whether it's usable and skip
# chemiscope later instead of hard-failing the whole script.
chemiscope_buildable=1
if ! command -v npm >/dev/null || ! command -v node >/dev/null; then
  echo "  MISSING: npm/node (only needed to build chemiscope)"
  chemiscope_buildable=0
elif [ "$(node -e 'console.log(process.versions.node.split(".")[0])')" -lt 20 ]; then
  echo "  npm/node present but node is too old for chemiscope (needs >=20): $(node --version)"
  chemiscope_buildable=0
fi

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

  # The driver caps the *newest* CUDA runtime it can run -- not just whether
  # it can run one at all. Installing PyTorch's default (newest bundled
  # CUDA) wheel on an older driver installs fine but then fails at runtime
  # with "CUDA initialization: The NVIDIA driver on your system is too old".
  # So read the driver's max supported CUDA version from `nvidia-smi`'s own
  # header and pick a wheel channel it can actually run.
  driver_cuda_ver=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
  index_url=""   # empty = default index = newest bundled CUDA build
  if [ -n "$driver_cuda_ver" ]; then
    echo "  Driver supports up to CUDA $driver_cuda_ver"
    index_url=$(awk -v v="$driver_cuda_ver" '
      BEGIN {
        if (v >= 12.8)      print ""
        else if (v >= 12.6) print "https://download.pytorch.org/whl/cu126"
        else if (v >= 12.4) print "https://download.pytorch.org/whl/cu124"
        else                print "https://download.pytorch.org/whl/cu121"
        # cu121 is the oldest channel PyTorch still publishes wheels for at
        # all; a driver older than that (< CUDA 11.8-ish) needs a newer
        # driver, there is no lower channel to fall back to.
      }')
  else
    echo "  Could not parse a CUDA version out of nvidia-smi -- using the default (newest) wheel"
  fi

  # No version floor here on purpose: PyTorch's *minimum* supported CUDA
  # version keeps rising with each release (e.g. cu121 tops out around
  # torch 2.5.x -- nothing newer ships a cu121 build at all), so pinning
  # e.g. "torch>=2.7" here makes an older-driver channel like cu121
  # unsatisfiable by construction. Let uv pick the newest torch actually
  # available on the chosen channel instead. If a specific package below
  # needs a torch floor (e.g. metatrain's dpa3 extra needs >=2.7), that
  # constraint belongs on that package's own extras, not hardcoded here.
  #
  # --reinstall-package is still required: `uv pip install "torch"` is a
  # no-op if any torch is already installed, even when pointing at a
  # different --index-url -- version satisfaction alone doesn't care which
  # CUDA build is actually present. Without forcing a reinstall, the
  # channel selection above would silently have no effect.
  if [ -n "$index_url" ]; then
    echo "  Installing PyTorch for CUDA <= $driver_cuda_ver ($index_url)"
    uv pip install --python "$VPY" --reinstall-package torch torch --index-url "$index_url"
  else
    echo "  Installing default (newest CUDA-enabled) PyTorch wheel"
    uv pip install --python "$VPY" --reinstall-package torch torch
  fi

  cuda_ok=$("$VPY" -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo False)
  if [ "$cuda_ok" != "True" ]; then
    echo "  WARNING: a driver is present but torch.cuda.is_available() is still False." >&2
    echo "  Re-run with a lower channel by hand, e.g.:" >&2
    echo "    uv pip install --python $VPY torch --index-url https://download.pytorch.org/whl/cu118" >&2
  fi
else
  echo "  No working NVIDIA driver found (this is the case on nouveau-only"
  echo "  boxes like this one) -- installing the smaller CPU-only wheel."
  echo "  Re-run this script unmodified on a machine with a real NVIDIA"
  echo "  driver to get CUDA-accelerated PyTorch instead."
  uv pip install --python "$VPY" torch --index-url https://download.pytorch.org/whl/cpu
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
  IFS=':' read -r repo extras _org <<< "$entry"
  if [ "$repo" = "chemiscope" ] && [ "$chemiscope_buildable" = 0 ]; then
    log "Skipping chemiscope (no usable npm/node -- see toolchain check above)"
    continue
  fi
  target="$BASE_DIR/$repo"
  [ -n "$extras" ] && target="$target[$extras]"
  log "Installing $repo${extras:+ [$extras]}"
  uv pip install --python "$VPY" -e "$target"
done

# ---- 5b. upet, in its own separate venv -------------------------------------
# upet (https://github.com/lab-cosmo/upet, universal PET-MAD/PET-OAM
# potentials) pins metatrain>=2026.4,<2026.5 -- a released version range our
# editable local `metatrain` checkout (a dev snapshot) does not satisfy.
# Installing upet into the shared venv works, but uv silently *replaces* the
# editable metatrain with the pinned PyPI release to satisfy it, which would
# stop picking up local edits to metatrain for everything else in the shared
# venv too. So upet gets its own venv instead, with its own pinned metatrain
# -- the shared venv's editable metatrain is left untouched.
UPET_VENV="$BASE_DIR/.venv-upet"
clone_or_update "upet" "lab-cosmo"

if [ ! -d "$UPET_VENV" ]; then
  log "Creating separate venv for upet at $UPET_VENV"
  uv venv "$UPET_VENV" --python "$PYTHON_VERSION"
fi
UPET_VPY="$UPET_VENV/bin/python"

log "Installing upet (separate venv)"
if [ -n "${index_url:-}" ]; then
  uv pip install --python "$UPET_VPY" --reinstall-package torch torch --index-url "$index_url"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  uv pip install --python "$UPET_VPY" --reinstall-package torch torch
else
  uv pip install --python "$UPET_VPY" --reinstall-package torch torch --index-url https://download.pytorch.org/whl/cpu
fi
uv pip install --python "$UPET_VPY" -e "$BASE_DIR/upet"

# ---- 6. summary --------------------------------------------------------------
log "Summary"
"$VPY" - <<'EOF'
import importlib
import torch

print(f"torch {torch.__version__}  (cuda build: {torch.version.cuda}, "
      f"cuda available at runtime: {torch.cuda.is_available()})")

for mod in ("metatensor", "metatensor.torch", "metatomic", "metatomic.torch", "featomic", "metatrain", "ipi", "chemiscope"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:20s} ok  ({getattr(m, '__version__', '')})")
    except Exception as e:
        print(f"{mod:20s} FAILED: {e}")
EOF

log "Summary (upet venv)"
"$UPET_VPY" - <<'EOF'
import importlib
import torch

print(f"torch {torch.__version__}  (cuda available at runtime: {torch.cuda.is_available()})")
for mod in ("metatrain", "upet"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:20s} ok  ({getattr(m, '__version__', '')})")
    except Exception as e:
        print(f"{mod:20s} FAILED: {e}")
EOF

echo
echo "Activate with:  source $VENV_DIR/bin/activate"
echo "Or run one-off: uv run --python $VPY <command>"
echo
echo "upet lives in its own venv (separate pinned metatrain, does not touch"
echo "the editable one above):"
echo "  Activate with:  source $UPET_VENV/bin/activate"
echo "  Or run one-off: uv run --python $UPET_VPY <command>"
