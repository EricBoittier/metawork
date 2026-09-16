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

# Intended checkout branch for each repo (submodule or independent clone).
# Keep this in sync with the `branch =` lines in .gitmodules. Empty / omitted
# means "leave whatever the clone defaulted to".
declare -A REPO_BRANCH=(
  [metatensor]="metatomic-core"
  [metatomic]="metatomic-core"
  [metatrain]="experimental/lorem"
  [featomic]="main"
  [i-pi]="main"
  [chemiscope]="main"
  [atomistic-cookbook]="metatomic-hourglass"
  [iris-infra]="main"
  [lorem-jax]="main"
  [upet]="main"
)

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
  "chemiscope::lab-cosmo"       # structure/property viewer; git submodule
                                 # from $FORK_OWNER/chemiscope (upstream
                                 # lab-cosmo). Build runs `npm` to bundle
                                 # JS assets -- see the toolchain check.
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

# True if $1 is listed as a git submodule of this repo (even if the
# checkout is still empty / not initialized).
is_submodule() {
  git -C "$BASE_DIR" config -f .gitmodules --get "submodule.$1.path" >/dev/null 2>&1
}

# Check out $2 in $1 if that branch exists on origin or upstream.
# No-op when the branch is empty, already checked out, or not found.
checkout_intended_branch() {
  local repo="$1"
  local branch="${2:-${REPO_BRANCH[$repo]:-}}"
  local dir="$BASE_DIR/$repo"
  [ -z "$branch" ] && return 0
  [ -e "$dir/.git" ] || return 0
  git -C "$dir" rev-parse --verify HEAD >/dev/null 2>&1 || return 0
  local current
  current="$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [ "$current" = "$branch" ]; then
    return 0
  fi
  log "Checking out $repo on $branch"
  git -C "$dir" fetch --all --quiet 2>/dev/null || true
  if git -C "$dir" show-ref --verify --quiet "refs/heads/$branch" \
     || git -C "$dir" show-ref --verify --quiet "refs/remotes/origin/$branch" \
     || git -C "$dir" show-ref --verify --quiet "refs/remotes/upstream/$branch"; then
    git -C "$dir" checkout "$branch" \
      || echo "  (could not checkout $branch on $repo -- leaving as-is)"
  else
    echo "  (no $branch branch on $repo -- leaving $current)"
  fi
}

# ---- 1. clone or update a repo, preferring your own fork -------------------
# org defaults to $UPSTREAM_ORG when empty (i.e. "" or omitted).
# After clone/update, checkout the intended branch from REPO_BRANCH.
clone_or_update() {
  local repo="$1"
  local org="${2:-$UPSTREAM_ORG}"
  local branch="${3:-${REPO_BRANCH[$repo]:-}}"
  [ -z "$org" ] && org="$UPSTREAM_ORG"
  # `.git` can be a file (gitlink for a submodule) or a directory
  # (independent clone). Either means the checkout already exists.
  if [ -e "$BASE_DIR/$repo/.git" ]; then
    # A checkout can be headless (no commit checked out) even after this
    # branch runs, e.g. from a clone interrupted before this self-healing
    # logic existed. `git pull` on a headless repo doesn't fix that, so
    # detect it and fall through to a fresh clone instead of just skipping.
    if git -C "$BASE_DIR/$repo" rev-parse --verify HEAD >/dev/null 2>&1; then
      checkout_intended_branch "$repo" "$branch"
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
  checkout_intended_branch "$repo" "$branch"
}

# Init a .gitmodules entry (or re-sync its URL), then checkout the
# intended branch. Falls back to an independent clone if submodule
# init fails. Adds an `upstream` remote pointing at $org when origin
# is a fork.
ensure_submodule() {
  local repo="$1"
  local org="${2:-$UPSTREAM_ORG}"
  local branch="${3:-${REPO_BRANCH[$repo]:-}}"
  [ -z "$org" ] && org="$UPSTREAM_ORG"
  # Legacy independent clone sitting in a path that is now also a
  # submodule -- leave it alone and just update it in place.
  if [ -d "$BASE_DIR/$repo/.git" ]; then
    clone_or_update "$repo" "$org" "$branch"
    return
  fi
  log "Initializing submodule $repo"
  git -C "$BASE_DIR" submodule sync -- "$repo" || true
  if ! git -C "$BASE_DIR" submodule update --init -- "$repo"; then
    echo "  (submodule init failed for $repo -- cloning independently)"
    clone_or_update "$repo" "$org" "$branch"
    return
  fi
  if ! git -C "$BASE_DIR/$repo" remote get-url upstream >/dev/null 2>&1; then
    git -C "$BASE_DIR/$repo" remote add upstream "https://github.com/$org/$repo.git" 2>/dev/null || true
  fi
  # chemiscope's webpack config runs `git describe --tags`. A submodule
  # clone of the fork has no tags (they live on lab-cosmo), so webpack
  # fails with "No names found, cannot describe anything".
  if [ "$repo" = "chemiscope" ]; then
    git -C "$BASE_DIR/$repo" fetch --tags upstream 2>/dev/null \
      || git -C "$BASE_DIR/$repo" fetch --tags origin 2>/dev/null \
      || echo "  (could not fetch chemiscope tags -- npm build may fail)"
  fi
  checkout_intended_branch "$repo" "$branch"
}

ensure_repo() {
  local repo="$1"
  local org="${2:-}"
  if is_submodule "$repo"; then
    ensure_submodule "$repo" "$org"
  else
    clone_or_update "$repo" "$org"
  fi
}

# metatensor-core/include/metatensor.h is regenerated by cbindgen as a side
# effect of building metatensor-core (see the metatensor-core pre-build step
# in the install loop below), which leaves it locally modified relative to
# whatever commit is checked out -- including across script runs, since a
# previous run's build (even one that failed at a later step) can leave this
# behind. `git checkout`/`git submodule update` then refuse to move the
# metatensor submodule to its pinned commit, aborting this whole step. The
# regenerated content carries no information the build doesn't reproduce
# identically next time, so it's always safe to discard here before
# touching any submodule state.
if [ -f "$BASE_DIR/metatensor/metatensor-core/include/metatensor.h" ]; then
  git -C "$BASE_DIR/metatensor" checkout -- metatensor-core/include/metatensor.h 2>/dev/null || true
fi

mkdir -p "$BASE_DIR"
for entry in "${INSTALL_REPOS[@]}"; do
  IFS=':' read -r repo _extras org <<< "$entry"
  ensure_repo "$repo" "$org"
done
for entry in "${CLONE_ONLY_REPOS[@]}"; do
  IFS=':' read -r repo org <<< "$entry"
  ensure_repo "$repo" "$org"
done

# Remaining .gitmodules entries that are not in INSTALL_REPOS /
# CLONE_ONLY_REPOS (iris-infra, lorem-jax, atomistic-cookbook). The
# recorded gitlink is the source of truth -- do not git pull them;
# just init and land on the intended branch.
if [ -f "$BASE_DIR/.gitmodules" ]; then
  log "Initializing remaining git submodules"
  git -C "$BASE_DIR" submodule sync
  # `|| true`: this fails (and would otherwise abort the whole script here,
  # under set -e) whenever featomic carries its usual uncommitted runtime
  # patches from a previous run (patch_featomic_*, above) -- by design we
  # never commit those since featomic is cloned straight from upstream, so
  # every run after the first leaves it "dirty" for this exact check. Other
  # submodules still update normally; checkout_intended_branch() right below
  # already tolerates this same failure case per-repo (see its "could not
  # checkout ... -- leaving as-is" branch).
  git -C "$BASE_DIR" submodule update --init || true
  for repo in "${!REPO_BRANCH[@]}"; do
    if [ -e "$BASE_DIR/$repo/.git" ]; then
      checkout_intended_branch "$repo" "${REPO_BRANCH[$repo]}"
    fi
  done
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
        # The ceiling bump alone (<0.11 -> <0.12) isn't enough: a locally-
        # built metatensor-torch==0.10.0.devNNN sorts *below* 0.10.0 by PEP
        # 440 (dev pre-releases precede their base version), so it still
        # fails the ">=0.10.0" floor even though it's effectively 0.10.x.
        # Lower the floor to 0.10.0.dev0 so 0.10.0 pre-releases satisfy it
        # (same fix as metatomic's own commit for the identical pin).
        "python/featomic_torch/build-backend/backend.py",
        'metatensor-torch >=0.10.0,<0.11',
        'metatensor-torch >=0.10.0.dev0,<0.12',
    ),
    (
        # Migrate a tree that already went through the old (ceiling-only) fix.
        "python/featomic_torch/build-backend/backend.py",
        'metatensor-torch >=0.10.0,<0.12',
        'metatensor-torch >=0.10.0.dev0,<0.12',
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
        'metatensor-torch >=0.10.0.dev0,<0.12',
    ),
    (
        # Migrate a tree that already went through the old (ceiling-only) fix.
        "python/featomic_torch/setup.py",
        'metatensor-torch >=0.10.0,<0.12',
        'metatensor-torch >=0.10.0.dev0,<0.12',
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
        # Our local metatensor-torch checkout is genuinely 0.10.x (its own
        # git-tag-derived dev version), so featomic-torch's original "0.10"
        # requirement was already correct -- an earlier version of this
        # patch wrongly bumped it to "0.11" (presumably valid for a
        # metatensor checkout further ahead at the time). Revert that.
        "featomic-torch/CMakeLists.txt",
        'set(REQUIRED_METATENSOR_TORCH_VERSION "0.11")',
        'set(REQUIRED_METATENSOR_TORCH_VERSION "0.10")',
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
    (
        # scripts/git-version-info.py's dirty-build path shells out to `git
        # write-tree` against a scratch copy of the *index file*, read
        # directly off disk as "ROOT/.git/index". That assumes ROOT/.git is
        # always a real directory -- true for a plain checkout, but featomic
        # is also cloned as a git submodule of the metawork superproject, in
        # which case ROOT/.git is a one-line gitlink *file* ("gitdir: ...")
        # and the real index lives under the superproject's
        # .git/modules/featomic/ instead. Opening "ROOT/.git/index" then
        # fails with NotADirectoryError, breaking every dirty (uncommitted-
        # changes) build -- which this ecosystem's own version-pin patches
        # above guarantee we always are. Ask git for the real git-dir
        # instead of assuming the on-disk layout.
        "scripts/git-version-info.py",
        '''        with tempfile.NamedTemporaryFile("wb") as tmp:
            with open(os.path.join(ROOT, ".git", "index"), "rb") as git_index:
                shutil.copyfileobj(git_index, tmp)
            tmp.close()''',
        '''        git_dir = run_subprocess(["git", "rev-parse", "--git-dir"]).stdout.strip()
        if not os.path.isabs(git_dir):
            git_dir = os.path.join(ROOT, git_dir)

        with tempfile.NamedTemporaryFile("wb") as tmp:
            with open(os.path.join(git_dir, "index"), "rb") as git_index:
                shutil.copyfileobj(git_index, tmp)
            tmp.close()''',
    ),
    (
        # Our local metatensor checkout (metatomic-core branch) renamed the
        # "create a TensorMap from a raw, owned mts_tensormap_t* without
        # checking it" escape hatch: the static `TensorMap::unsafe_from_ptr`
        # factory (C++) / `TensorMap.unsafe_from_ptr` classmethod (Python)
        # became a plain `explicit TensorMap(mts_tensormap_t*)` constructor
        # (C++) / `TensorMap._from_ptr` (Python) instead. featomic's own
        # C++ header and Python bindings still call the old names, so
        # linking against our local metatensor breaks with "'unsafe_from_ptr'
        # is not a member of 'metatensor::TensorMap'". Same semantics, just
        # renamed -- swap the call sites.
        "featomic/include/featomic.hpp",
        "metatensor::TensorMap::unsafe_from_ptr(descriptor)",
        "metatensor::TensorMap(descriptor)",
    ),
    (
        "python/featomic/featomic/calculator_base.py",
        "TensorMap.unsafe_from_ptr(tensor_map_ptr)",
        "TensorMap._from_ptr(tensor_map_ptr)",
    ),
]

import os
for rel_path, old, new in patches:
    path = os.path.join(root, rel_path)
    if not os.path.isfile(path):
        continue
    text = open(path).read()
    if old in text:
        # replace every occurrence (e.g. the unsafe_from_ptr rename above
        # appears twice in featomic.hpp) -- not just the first.
        open(path, "w").write(text.replace(old, new))
        print("  patched", path)
PYEOF
}
patch_featomic_metatensor_version_pins

# ---- 1c. patch metatomic's stale metatensor-core version pins -------------
# Same problem as patch_featomic_metatensor_version_pins above, one repo
# over: metatomic-core's own pins were written against the last released
# metatensor-core (0.2.x); our local editable metatensor checkout has since
# moved ahead to 0.3.x-dev, which these pins reject outright.
#   - the Python-level `metatensor-core >=0.2.4,<0.3` pin makes
#     `uv pip install -e metatomic[torch]` unsatisfiable against the local
#     0.3.0.dev... build.
#   - the CMake-level `find_package(metatensor 0.2.4 ...)` call uses
#     same-minor-version compatibility for 0.x releases, so it rejects the
#     installed 0.3 config outright at build time (same pattern fixed in
#     featomic's CMakeLists.txt).
#   - metatomic-torch's own `metatensor-torch >=0.10.0,<0.11` pin has the
#     same "dev pre-release sorts below its base version" floor problem
#     already fixed in featomic-torch: a locally-built
#     metatensor-torch==0.10.0.devNNN doesn't satisfy a bare ">=0.10.0".
# No-op for any file/line upstream has already updated, or that we already
# patched on a previous run.
patch_metatomic_metatensor_version_pins() {
  local root="$BASE_DIR/metatomic"
  [ -d "$root" ] || return 0
  python3 - "$root" <<'PYEOF'
import sys

root = sys.argv[1]

# (file relative to metatomic/, old substring, new substring)
patches = [
    (
        "python/metatomic_core/pyproject.toml",
        "metatensor-core >=0.2.4,<0.3",
        "metatensor-core >=0.2.4,<0.4",
    ),
    (
        # pyproject.toml's dependencies are static leftovers -- the pin
        # setup.py's dynamic-metadata hook actually emits (and what uv
        # resolves against) is this one, in setup.py itself. Same ceiling
        # bump as above, just a different floor value (0.2.2 vs 0.2.4).
        "python/metatomic_core/setup.py",
        "metatensor-core >=0.2.2,<0.3",
        "metatensor-core >=0.2.2,<0.4",
    ),
    (
        # metatomic/_c_api.py unconditionally does
        # `from ctypes_dlpack import ...`, but metatomic-core's own
        # install_requires never lists the `ctypes-dlpack` PyPI package
        # that module comes from -- an upstream missing-dependency bug.
        # Surfaces as `ModuleNotFoundError: No module named 'ctypes_dlpack'`
        # the moment anything imports metatomic (e.g. featomic-torch's
        # setup.py, which imports metatomic at build time to read its C API).
        "python/metatomic_core/setup.py",
        '''    install_requires = [
        "metatensor-core >=0.2.2,<0.4",
    ]''',
        '''    install_requires = [
        "metatensor-core >=0.2.2,<0.4",
        "ctypes-dlpack",
    ]''',
    ),
    (
        "metatomic-core/CMakeLists.txt",
        'set(REQUIRED_METATENSOR_VERSION "0.2.4")',
        'set(REQUIRED_METATENSOR_VERSION "0.3")',
    ),
    (
        "python/metatomic_torch/pyproject.toml",
        "metatensor-torch >=0.10.0,<0.11",
        "metatensor-torch >=0.10.0.dev0,<0.12",
    ),
    (
        "python/metatomic_torch/setup.py",
        "metatensor-torch >=0.10.0,<0.11",
        "metatensor-torch >=0.10.0.dev0,<0.12",
    ),
    (
        "python/metatomic_torch/setup.py",
        "metatensor-operations >=0.5.0,<0.6",
        "metatensor-operations >=0.5.0,<0.7",
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
patch_metatomic_metatensor_version_pins

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

# ---- 4b. CUDA toolkit (nvcc) detection, incl. host-compiler compatibility --
# metatensor-torch / metatomic-torch / featomic compile actual CUDA kernels
# at build time, which needs a *working* `nvcc` -- not just one on PATH. See
# ensure_working_cuda_toolchain() in _torch_cuda.sh for why "nvcc is on
# PATH" alone is not a sufficient check on this kind of machine.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  log "Checking for a working CUDA toolkit (nvcc + compatible host compiler)"
  ensure_working_cuda_toolchain || true
fi

# ---- 5. install the ecosystem packages, in dependency order ----------------
# Each repo's install is independent of the others actually *succeeding*
# (only its presence on disk, handled by ensure_repo() above, matters --
# e.g. metatrain[soap-bpnn,pet] does not depend on featomic at all, even
# though featomic is installed as its own earlier entry here). So a build
# failure in one repo (tracked in INSTALL_FAILED) is logged and skipped
# rather than aborting the whole script and leaving every later repo
# uninstalled -- that would otherwise turn one broken package into a
# reason the *entire* ecosystem fails to install on a fresh node.
INSTALL_FAILED=()
INSTALL_OK=()

install_one_repo() {
  local repo="$1" extras="$2"
  local target="$BASE_DIR/$repo"
  [ -n "$extras" ] && target="$target[$extras]"

  if [ "$repo" = "metatensor" ]; then
    # metatensor-core's own build regenerates metatensor-core/include/
    # metatensor.h via cbindgen (its content differs from the committed
    # version even right after a fresh, clean submodule checkout). uv builds
    # metatensor-core and metatensor-torch from this same invocation's
    # dependency graph in parallel; metatensor-torch's setup.py computes its
    # own version by shelling out to `git diff-index --quiet HEAD --`
    # (scripts/git-version-info.py) to decide the "dirty."/"git." version
    # suffix, once for the wheel filename and once for the wheel metadata.
    # If metatensor-core's build regenerates the header in between those two
    # checks, one reads "clean" and the other "dirty", and uv rejects the
    # wheel with "Package metadata version ... does not match ... from the
    # wheel filename". Pre-building metatensor-core on its own first lets
    # that regeneration happen and settle *before* metatensor-torch's build
    # starts, so both of its git-dirty checks see the same (dirty) state.
    log "Pre-building metatensor-core (avoids a git-dirty race with metatensor-torch's parallel build)"
    write_local_pkgs_constraint "$VPY"
    uv_pip_keep_torch "$VPY" "$BASE_DIR/metatensor/python/metatensor_core" || return 1
  fi

  if [ "$repo" = "featomic" ]; then
    # Same intra-package build-ordering race as metatensor-core above, one
    # level up the stack: featomic-torch's CMake build finds featomic via
    # its installed featomic-config.cmake, which embeds whatever
    # METATENSOR_REQUIRED_VERSION featomic/CMakeLists.txt was built with.
    # patch_featomic_metatensor_version_pins() (section 1b, above) already
    # bumps that to "0.3" in the source tree, but if featomic-torch's build
    # (from this same combined install) starts before featomic's own build
    # has finished and reinstalled, it can still find a *stale* previously-
    # installed featomic-config.cmake pinned to the old "0.2". Pre-building
    # featomic on its own first guarantees the freshly-patched config is in
    # place before featomic-torch's CMake configure step reads it.
    log "Pre-building featomic (avoids a stale featomic-config.cmake race with featomic-torch's parallel build)"
    write_local_pkgs_constraint "$VPY"
    uv_pip_keep_torch "$VPY" "$BASE_DIR/featomic/python/featomic" || return 1
  fi

  log "Installing $repo${extras:+ [$extras]}"
  # Refresh right before each install: a repo installed earlier in this loop
  # (e.g. metatensor) may be depended on by name from one installed later
  # (e.g. metatomic) -- see write_local_pkgs_constraint in _torch_cuda.sh.
  write_local_pkgs_constraint "$VPY"
  uv_pip_keep_torch "$VPY" -e "$target"
}

for entry in "${INSTALL_REPOS[@]}"; do
  IFS=':' read -r repo extras _org <<< "$entry"
  if [ "$repo" = "chemiscope" ] && [ "$chemiscope_buildable" = 0 ]; then
    log "Skipping chemiscope (no usable npm/node -- see toolchain check above)"
    continue
  fi
  if install_one_repo "$repo" "$extras"; then
    INSTALL_OK+=("$repo${extras:+ [$extras]}")
  else
    INSTALL_FAILED+=("$repo${extras:+ [$extras]}")
    echo "  (continuing -- $repo failed to install, see error above; other repos do not depend on it succeeding unless they list it as a real dependency)" >&2
  fi
done

log "Ecosystem install summary"
if [ "${#INSTALL_OK[@]}" -gt 0 ]; then
  printf '  OK:     %s\n' "${INSTALL_OK[@]}"
fi
if [ "${#INSTALL_FAILED[@]}" -gt 0 ]; then
  printf '  FAILED: %s\n' "${INSTALL_FAILED[@]}" >&2
fi

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
clone_or_update "upet" "lab-cosmo" "main"

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

if [ "${#INSTALL_FAILED[@]}" -gt 0 ]; then
  echo
  echo "NOTE: ${#INSTALL_FAILED[@]} repo(s) failed to install (see 'Ecosystem" >&2
  echo "install summary' above for which, and the build output further up for" >&2
  echo "why); everything else still installed. Exiting non-zero so this is" >&2
  echo "visible in CI/automation even though the run otherwise completed." >&2
  exit 1
fi
