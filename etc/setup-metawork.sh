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
    # A checkout can be headless (no commit checked out) even after this
    # branch runs, e.g. from a clone interrupted before this self-healing
    # logic existed. `git pull` on a headless repo doesn't fix that, so
    # detect it and fall through to a fresh clone instead of just skipping.
    if git -C "$BASE_DIR/$repo" rev-parse --verify HEAD >/dev/null 2>&1; then
      log "Updating $repo"
      git -C "$BASE_DIR/$repo" pull --ff-only \
        || echo "  (skipped: local changes or diverged branch -- update $repo by hand)"
      return
    fi
    echo "  $repo has no commit checked out (broken/interrupted clone) -- re-cloning"
    rm -rf "$BASE_DIR/$repo"
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

# iris-infra and lorem-jax are tracked git submodules (JAX reference
# implementations), not INSTALL_REPOS / CLONE_ONLY_REPOS checkouts. The
# recorded gitlink is the source of truth -- do not git pull them.
if [ -f "$BASE_DIR/.gitmodules" ]; then
  log "Initializing git submodules"
  git -C "$BASE_DIR" submodule update --init
fi

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

# featomic is cloned straight from upstream (metatensor/featomic) rather
# than from a fork of ours, so unlike metatomic/metatensor we can't fix
# this by committing to it -- patch it in place instead, same as the cudnn
# bug above. featomic's own pins on metatensor-core / metatensor-torch
# were written against the last released versions (0.2.x / 0.10.x); our
# local editable metatensor checkout has since moved ahead to 0.3.x-dev /
# 0.11.x-dev, which these pins reject outright:
#   - the Python-level `metatensor-core >=0.2.2,<0.3` / `metatensor-torch
#     >=0.10.0,<0.11` pins make `uv pip install -e featomic[torch]`
#     unsatisfiable against the local dev versions.
#   - the CMake-level `find_package(metatensor 0.2 ...)` / `find_package
#     (metatensor_torch 0.10 ...)` calls use same-minor-version
#     compatibility for 0.x releases, so they reject the installed 0.3 /
#     0.11 configs outright at build time (see the same pattern fixed in
#     metatomic's CMakeLists.txt).
# No-op for any file/line upstream has already updated, or that we already
# patched on a previous run.
patch_featomic_metatensor_version_pins() {
  local root="$BASE_DIR/featomic"
  [ -d "$root" ] || return 0
  python3 - "$root" <<'PYEOF'
import sys

root = sys.argv[1]

# (file relative to featomic/, old substring, new substring)
patches = [
    (
        "python/featomic/pyproject.toml",
        "metatensor-core >=0.2.2,<0.3",
        "metatensor-core >=0.2.2,<0.4",
    ),
    (
        "python/featomic/pyproject.toml",
        "metatensor-operations >=0.5.0,<0.6",
        "metatensor-operations >=0.5.0,<0.7",
    ),
    (
        "python/featomic/build-backend/backend.py",
        'metatensor-core >=0.2.2,<0.3',
        'metatensor-core >=0.2.2,<0.4',
    ),
    (
        "python/featomic_torch/build-backend/backend.py",
        'metatensor-torch >=0.10.0,<0.11',
        'metatensor-torch >=0.10.0,<0.12',
    ),
    (
        # 0.2.0.dev-prerelease local builds of metatomic-torch don't satisfy
        # an exclusive "<0.2" bound: PEP 440 excludes *all* pre-releases of
        # the excluded boundary version itself (0.2.0.devN < 0.2 numerically,
        # but a bare "<0.2" with no pre-release marker on 0.2 still rejects
        # it, by design -- see the SpecifierSet docs).
        "python/featomic_torch/build-backend/backend.py",
        'metatomic-torch >=0.1.15,<0.2',
        'metatomic-torch >=0.1.15,<0.3',
    ),
    (
        # Same two pins as above, duplicated in setup.py's own
        # install_requires (the actual wheel metadata, computed separately
        # from build-backend.py's build-time-only requirements).
        "python/featomic_torch/setup.py",
        'metatensor-torch >=0.10.0,<0.11',
        'metatensor-torch >=0.10.0,<0.12',
    ),
    (
        "python/featomic_torch/setup.py",
        'metatomic-torch >=0.1.15,<0.2',
        'metatomic-torch >=0.1.15,<0.3',
    ),
    (
        "featomic/CMakeLists.txt",
        'set(METATENSOR_REQUIRED_VERSION "0.2")',
        'set(METATENSOR_REQUIRED_VERSION "0.3")',
    ),
    (
        "featomic-torch/CMakeLists.txt",
        'set(REQUIRED_METATENSOR_TORCH_VERSION "0.10")',
        'set(REQUIRED_METATENSOR_TORCH_VERSION "0.11")',
    ),
    (
        "featomic-torch/CMakeLists.txt",
        'set(REQUIRED_METATOMIC_TORCH_VERSION "0.1")',
        'set(REQUIRED_METATOMIC_TORCH_VERSION "0.2")',
    ),
    (
        # scripts/git-version-info.py computes the build version from git,
        # but its subprocess calls never pin cwd -- they rely on inheriting
        # a CWD already inside the repo. That breaks under some PEP517
        # build backends (e.g. setuptools' _build_with_temp_dir) that chdir
        # elsewhere before running this code, so every git call fails with
        # "not a git repository" (exit 128, empty stdout/stderr). Pin
        # cwd=ROOT explicitly on both the shared run_subprocess() helper
        # and the one raw subprocess.run() call that bypasses it.
        "scripts/git-version-info.py",
        '''    output = subprocess.run(
        args,
        capture_output=True,
        encoding="utf8",
        check=False,
        env=env,
    )''',
        '''    output = subprocess.run(
        args,
        capture_output=True,
        encoding="utf8",
        check=False,
        env=env,
        cwd=ROOT,
    )''',
    ),
    (
        "scripts/git-version-info.py",
        '''    output = subprocess.run(
        ["git", "diff-index", "--quiet", "HEAD", "--"],
        capture_output=True,
    )''',
        '''    output = subprocess.run(
        ["git", "diff-index", "--quiet", "HEAD", "--"],
        capture_output=True,
        cwd=ROOT,
    )''',
    ),
]

import os
for rel_path, old, new in patches:
    path = os.path.join(root, rel_path)
    if not os.path.isfile(path):
        continue
    text = open(path).read()
    if old in text:
        open(path, "w").write(text.replace(old, new, 1))
        print("  patched", path)
PYEOF
}
patch_featomic_metatensor_version_pins

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
# shellcheck source=etc/_torch_cuda.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_torch_cuda.sh"

# ---- 4. GPU-aware PyTorch install -------------------------------------------
log "Detecting GPU / CUDA driver"
# metatrain's soap-bpnn extra depends on sphericart-torch, which pins
# torch>=2.6,<2.14 -- without this, ensure_torch_for_driver just grabs the
# newest torch that initializes CUDA on this driver (currently 2.14.0),
# which is one release past sphericart-torch's ceiling and makes metatrain
# [soap-bpnn] unsatisfiable. Remove/adjust this once sphericart-torch
# supports newer torch.
TORCH_VERSION_CONSTRAINT="torch>=2.6,<2.14"
ensure_torch_for_driver "$VPY" || true
# uv_pip_keep_torch below builds every [torch]-extra package with
# --no-build-isolation (see _torch_cuda.sh) to keep both the build backend
# and the torch it links against pinned to what's already in the venv --
# seed those build-time packages now since `uv venv` doesn't install them.
ensure_build_seed_packages "$VPY"

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
  # Refresh right before each install: a repo installed earlier in this loop
  # (e.g. metatensor) may be depended on by name from one installed later
  # (e.g. metatomic) -- see write_local_pkgs_constraint in _torch_cuda.sh.
  write_local_pkgs_constraint "$VPY"
  uv_pip_keep_torch "$VPY" -e "$target"
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
TORCH_PIN_FILE="$UPET_VENV/.torch-constraint.txt"
LOCAL_PKGS_CONSTRAINT_FILE="$UPET_VENV/.local-pkgs-constraint.txt"
ensure_torch_for_driver "$UPET_VPY" || true
ensure_build_seed_packages "$UPET_VPY"
write_local_pkgs_constraint "$UPET_VPY"
uv_pip_keep_torch "$UPET_VPY" -e "$BASE_DIR/upet"
TORCH_PIN_FILE="$VENV_DIR/.torch-constraint.txt"
LOCAL_PKGS_CONSTRAINT_FILE="$VENV_DIR/.local-pkgs-constraint.txt"

# ---- 5c. extra PyPI packages for etc/ examples ------------------------------
# Not part of the ecosystem checkouts. Used by etc/qm7x_zenodo (HDF5).
# Safe to re-run -- uv is a no-op if the version is already satisfied.
log "Installing extra example dependencies"
uv_pip_keep_torch "$VPY" h5py

# ---- 5d. keep the driver-matched torch --------------------------------------
# Editable `[torch]` extras resolve `torch` from PyPI and will swap a cu121
# wheel for 2.13+cu130 (same version constraint, different CUDA runtime).
# Re-check after those installs; if CUDA died, restore the matching wheel
# and rebuild the extension packages against it.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 \
   && ! torch_cuda_is_available "$VPY"; then
  log "torch.cuda.is_available() is False after package installs -- restoring a driver-matched wheel"
  torch_before=$("$VPY" -c "import torch; print(torch.__version__)")
  ensure_torch_for_driver "$VPY" || true
  torch_after=$("$VPY" -c "import torch; print(torch.__version__)")
  if [ "$torch_before" != "$torch_after" ]; then
    log "torch changed $torch_before -> $torch_after; rebuilding torch extension packages"
    for entry in "${INSTALL_REPOS[@]}"; do
      IFS=':' read -r repo extras _org <<< "$entry"
      [ "$extras" = "torch" ] || continue
      log "Reinstalling $repo[torch]"
      uv_pip_keep_torch "$VPY" --reinstall-package "${repo}-torch" -e "$BASE_DIR/$repo[$extras]"
    done
  fi
fi

# ---- 6. summary --------------------------------------------------------------
log "Summary"
"$VPY" - <<'EOF'
import importlib
import torch

print(f"torch {torch.__version__}  (cuda build: {torch.version.cuda}, "
      f"cuda available at runtime: {torch.cuda.is_available()})")

for mod in (
    "metatensor", "metatensor.torch", "metatomic", "metatomic.torch",
    "featomic", "metatrain", "ipi", "chemiscope", "h5py",
):
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
