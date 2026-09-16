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

# The ecosystem packages (metatensor-torch, metatomic-torch, featomic-torch)
# each build against and pin to `torch=={the major.minor of whatever torch
# is present in their own PEP517 build isolation}`, and that build-time
# resolve isn't scoped to our own local editable checkouts -- a package's
# own `build-system.requires` pulls in the latest matching PyPI *release*
# of the others (e.g. metatomic-torch's pyproject.toml requires
# `metatensor-torch >=0.10.0,<0.11`), which itself was published pinned to
# whatever torch was newest *then*. So the moment PyPI ships a newer torch
# than these upstream releases were built against, `uv pip install -e
# metatomic[torch]` becomes unsatisfiable: our locally-built
# metatensor-torch chases the newest torch (no pin holds it back), while
# metatomic-torch's build drags in an old PyPI metatensor-torch pinned to
# an older one -- see README Known Issues for the full story. A plain
# constraints file (`-c ...`) does not help: constraints only scope the
# top-level install graph, not each package's own isolated build
# resolution.
#
# `--exclude-newer-package` fixes this at the resolver level instead: it
# hides `torch` releases newer than this date from *every* index lookup,
# including ones happening inside another package's own build isolation,
# so every package converges on the same, older, mutually-compatible torch
# instead of each independently chasing "newest available today". Bump (or
# remove) this once upstream republishes against the newer torch.
TORCH_EXCLUDE_NEWER_DATE="2026-09-05"
TORCH_EXCLUDE_NEWER_ARGS=(--exclude-newer-package "torch=$TORCH_EXCLUDE_NEWER_DATE")

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

# Optional version ceiling/floor on top of whatever channel-walk picks the
# CUDA-compatible build -- e.g. "torch<2.14" when some extra we install
# (metatrain's soap-bpnn, via sphericart-torch) caps torch below the latest
# release. Empty (default) = no extra constraint, just "torch".
TORCH_VERSION_CONSTRAINT="${TORCH_VERSION_CONSTRAINT-}"

install_torch_from_index() {
  local py="$1"
  local url="${2-}"
  local spec="${TORCH_VERSION_CONSTRAINT:-torch}"
  if [ -n "$url" ]; then
    echo "  uv pip install $spec --index-url $url"
    uv pip install --python "$py" --reinstall-package torch "$spec" --index-url "$url"
  else
    echo "  uv pip install $spec (PyPI default CUDA wheel)"
    uv pip install --python "$py" --reinstall-package torch "$spec"
  fi
}

pin_torch() {
  local py="$1"
  local dest="${TORCH_PIN_FILE:-${VENV_DIR:-.}/.torch-constraint.txt}"
  mkdir -p "$(dirname "$dest")"
  # Pin the *public* version only (e.g. "2.13.0"), not the full local
  # version PyPI's own CUDA wheels carry (e.g. "2.13.0+cu130"). uv/pip
  # constraints are resolved against index metadata, and a local version
  # segment never appears there -- constraining to the exact local version
  # makes every later `-c`/`-b` install that needs to *resolve* torch (any
  # editable/local package with a plain `torch` dependency) fail with "there
  # is no version of torch==X+cuYYY", even though that exact wheel is
  # already installed. Per PEP 440, a bare "torch==2.13.0" constraint still
  # matches the installed "2.13.0+cu130" (a public-version-only specifier
  # matches any local version of it), so this keeps the CUDA build pinned
  # without breaking resolution.
  "$py" -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > "$dest"
  echo "  pinned $("$py" -c 'import torch; print(torch.__version__)') -> $dest"
}

# Install a torch build this driver can initialize. Starts at the channel
# matching nvidia-smi's CUDA Version, then walks older channels if
# torch.cuda.is_available() is still False (nvidia-smi header can be ahead
# of what the running driver actually supports).
ensure_torch_for_driver() {
  local py="$1"

  # Building on a node with no local GPU (e.g. an HPC login node) for a
  # driver elsewhere (e.g. a GPU compute node) -- there's nothing here for
  # nvidia-smi/torch.cuda.is_available() to probe, so the channel walk below
  # can't run. Set TORCH_FORCE_INDEX_URL to skip detection and use that
  # channel directly, unverified (empty string = PyPI default wheel).
  if [ -n "${TORCH_FORCE_INDEX_URL+x}" ]; then
    echo "  TORCH_FORCE_INDEX_URL set -- installing torch from '${TORCH_FORCE_INDEX_URL:-PyPI default}' without driver detection"
    if ! install_torch_from_index "$py" "$TORCH_FORCE_INDEX_URL"; then
      echo "  ERROR: forced torch install from '${TORCH_FORCE_INDEX_URL:-PyPI default}' failed -- not pinning, leaving whatever torch was already in the venv untouched" >&2
      return 1
    fi
    TORCH_INDEX_URL="$TORCH_FORCE_INDEX_URL"
    pin_torch "$py"
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "  No working NVIDIA driver -- installing CPU torch"
    uv pip install --python "$py" --reinstall-package torch \
      --index-url https://download.pytorch.org/whl/cpu \
      "${TORCH_EXCLUDE_NEWER_ARGS[@]}" \
      "${TORCH_VERSION_CONSTRAINT:-torch}"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    pin_torch "$py"
    return 0
  fi

  echo "  Working NVIDIA driver detected:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/    /'

  local ver mapped ch
  ver="$(driver_max_cuda)"
  mapped="$(index_url_for_driver_cuda "${ver:-0}")"
  echo "  Driver reports CUDA ${ver:-unknown} (nvidia-smi's naive per-version guess: ${mapped:-PyPI default})"

  # nvidia-smi's "CUDA Version" is not a hard ceiling: NVIDIA's minor-
  # version compatibility within a CUDA major series often lets a driver
  # run torch wheels built for a *newer* CUDA minor than the one reported
  # (confirmed in practice: a driver reporting "CUDA Version: 12.2" here
  # initializes a cu126-built torch fine, even though index_url_for_driver
  # _cuda's naive mapping would have stopped at cu121 -- which, in turn,
  # stopped shipping torch releases after 2.5.1, silently capping every
  # package in this ecosystem to an increasingly old torch forever). So
  # always try the *entire* channel list newest-first and empirically keep
  # the first one that actually initializes CUDA, rather than trusting the
  # naive mapping as a starting point -- it's index_url_for_driver_cuda's
  # only remaining job to print context above, not to restrict this walk.
  for ch in "${TORCH_CUDA_CHANNELS[@]}"; do
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

# `--no-build-isolation` (see uv_pip_keep_torch) needs setuptools/wheel/a
# recent-enough packaging already present in the venv -- `uv venv` doesn't
# seed those by default. Call this once per venv before the first
# uv_pip_keep_torch install.
#
# setuptools_scm is here for the same reason, one level more subtle:
# --no-build-isolation means uv never installs a package's own
# `build-system.requires` at all, it just reuses whatever's already in the
# venv. A package that's declared `dynamic = ["version"]` via
# `[tool.setuptools_scm]` (e.g. metatrain) only lists setuptools_scm in
# build-system.requires, never in its runtime install_requires -- so
# without it pre-seeded here, its build silently falls back to a
# 0.0.0/no-op version instead of failing loudly, *and* skips writing the
# `_version.py` file the package's own `__init__.py` imports, breaking the
# import outright (`ModuleNotFoundError: No module named
# 'metatrain._version'`) rather than at build time where it'd be obvious.
ensure_build_seed_packages() {
  local py="$1"
  uv pip install --python "$py" -U "packaging>=24.2" setuptools wheel "setuptools_scm>=8"
}

# When one editable-installed local package (e.g. metatomic-torch) depends
# by name on another that's *also* installed from a local path (e.g.
# metatensor-core, metatensor-operations), uv refuses to resolve it unless
# that local source is stated explicitly as a direct requirement or
# constraint -- otherwise it tries to satisfy the dependency from PyPI,
# which can be a much older release than the local dev version and fail
# the resolve outright ("URL dependencies must be expressed as direct
# requirements or constraints", or an unrelated-looking version conflict
# against a stale PyPI release). This is a separate problem from the
# torch-drift one above -- --exclude-newer-package does not touch it, since
# it's not about release dates, it's about uv refusing to substitute a
# local path for a plain-name PyPI dependency without being told to.
# Regenerate this before each install in the dependency-ordered loop, since
# later repos depend on earlier ones.
write_local_pkgs_constraint() {
  local py="$1"
  local dest="${LOCAL_PKGS_CONSTRAINT_FILE:-${VENV_DIR:-.}/.local-pkgs-constraint.txt}"
  mkdir -p "$(dirname "$dest")"
  # Only named "pkg @ file://..." lines are valid as constraints; `uv pip
  # freeze` also emits unnamed "-e file://..." lines for the top-level
  # package of each editable checkout, which uv rejects here -- drop those.
  uv pip freeze --python "$py" \
    | grep -E '^[A-Za-z0-9_.-]+ @ file://' \
    > "$dest" || true
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
    # A `-c` constraints file only scopes the top-level install graph, not
    # the separate isolated build environment each package's own
    # `build-system.requires` resolves for itself -- so on its own it does
    # not stop e.g. metatensor-torch's build from picking a newer `torch`
    # than what we just pinned above (in testing, --exclude-newer-package
    # alone still let an isolated build resolve an established-but-newer
    # release, since "newer than the exclude-newer date" and "newer than
    # our pin" are different cutoffs). `-b`/--build-constraints is the
    # flag that actually reaches inside build isolation, so pass the same
    # pin there too to force every package's build to use our exact torch.
    args+=(-b "$pin")
  fi
  local local_pkgs="${LOCAL_PKGS_CONSTRAINT_FILE:-${VENV_DIR:-.}/.local-pkgs-constraint.txt}"
  if [ -f "$local_pkgs" ]; then
    args+=(-c "$local_pkgs")
    # Same reasoning as -b "$pin" above: a package being built in isolation
    # (e.g. featomic's CMakeLists.txt find_package(metatensor ...)) can
    # depend by name on another local-editable package (metatensor) without
    # that isolated build seeing our local checkout at all -- it resolves
    # its own copy from PyPI instead (observed: a stale metatensor==0.2.4
    # wheel, while the venv already has a newer local dev build installed).
    # -b reaches inside that isolated build the same way -c reaches the
    # top-level graph, forcing it to use the exact same local sources.
    args+=(-b "$local_pkgs")
  fi
  # --exclude-newer-package still matters alongside --build-constraints
  # above: it keeps *other* build-time resolutions (e.g. packaging, or a
  # sibling package's own transitive deps) from drifting to a newer release
  # than what everything else in the ecosystem was validated against, in
  # index lookups build-constraints doesn't cover. See the
  # TORCH_EXCLUDE_NEWER_DATE comment above.
  args+=("${TORCH_EXCLUDE_NEWER_ARGS[@]}")
  # --no-build-isolation: even with -b/--build-constraints pinning *which*
  # local metatensor-torch a nested build resolves, that's not enough for
  # packages whose build reads another package's runtime state rather than
  # just its declared version -- metatensor_torch.utils.cmake_prefix_path
  # picks its ABI subdirectory (torch-2.5 vs torch-2.14, it ships several
  # side by side) from `torch.__version__` *in the process actually running
  # the build*, so a separate isolated build env with its own, differently-
  # resolved torch silently points CMake at the wrong one (observed: CMake
  # error "metatensor-torch was built against v2.5.1 but we found v2.14.0",
  # even though every constraint above was satisfied). --no-build-isolation
  # removes the separate env entirely, so the build always sees exactly the
  # torch (and every local editable package) already in this venv.
  #
  # --refresh: uv's build cache for local/editable sources is keyed on the
  # source tree, not on which torch/index config was active when it was
  # last built -- a wheel built and cached from an earlier, differently-
  # pinned run (e.g. back when torch had drifted, before the fixes above
  # existed) can otherwise get silently reused even though none of them
  # would produce that same wheel today. metatomic-torch's setup.py (and
  # similar packages) reads torch.__version__ at build time and bakes an
  # exact `torch == X.Y.*` pin into its own install-requires, so a stale
  # cached wheel here surfaces as a confusing version-resolution conflict
  # rather than an obviously-stale-cache symptom.
  uv pip install --python "$py" --no-build-isolation --refresh "${args[@]}" "$@"
}

# nvcc enforces a maximum supported host-compiler (gcc/g++) version tied to
# the CUDA toolkit release (e.g. CUDA 12.4 rejects gcc >13). Distros move
# their default gcc forward far faster than CUDA toolkits add support for
# it -- this machine ships gcc 15/16 by default -- so "nvcc is on PATH" is
# not enough: nvcc can be found and still fail every real CUDA build with
# "error: unsupported GNU version". Detect that with an actual trivial-
# kernel compile, and if it fails this way, get a compatible g++ from
# conda (conda-forge ships every gcc major version, unlike most distro
# repos, which drop old gcc quickly) and point CUDAHOSTCXX -- CMake's own
# documented env var for CMAKE_CUDA_HOST_COMPILER -- at it. Cheaper and
# more portable than installing a second system-wide gcc via the package
# manager, and needs no sudo.
#
# Only matters for the main ecosystem venv: metatensor/metatomic/featomic
# compile real CUDA kernels from source here, while `upet` (a separate
# venv) only ever pulls prebuilt metatensor-torch/metatomic-torch wheels
# from PyPI, so it never invokes nvcc at all.
# Compiles a trivial .cu file with nvcc, optionally via -ccbin (mirroring
# how CMake's CUDAHOSTCXX-initialized CMAKE_CUDA_HOST_COMPILER actually
# invokes nvcc -- nvcc itself does not read the CUDAHOSTCXX env var, so a
# bare `nvcc` probe would silently ignore it and always test the default
# host compiler).
_probe_nvcc_compile() {
  local ccbin="${1:-}" test_dir out status
  test_dir="$(mktemp -d)"
  cat > "$test_dir/probe.cu" <<'EOF'
__global__ void k() {}
int main() { k<<<1,1>>>(); return 0; }
EOF
  if [ -n "$ccbin" ]; then
    out="$(nvcc -ccbin "$ccbin" -o "$test_dir/probe" "$test_dir/probe.cu" 2>&1)"
  else
    out="$(nvcc -o "$test_dir/probe" "$test_dir/probe.cu" 2>&1)"
  fi
  status=$?
  rm -rf "$test_dir"
  printf '%s' "$out"
  return "$status"
}

ensure_cuda_host_compiler() {
  command -v nvcc >/dev/null 2>&1 || return 0
  [ -n "${CUDAHOSTCXX:-}" ] && return 0

  local out status
  out="$(_probe_nvcc_compile)"
  status=$?
  [ "$status" = 0 ] && return 0

  local max_gcc
  max_gcc="$(printf '%s' "$out" | grep -oE 'gcc versions later than [0-9]+' | grep -oE '[0-9]+' | head -1)"
  if [ -z "$max_gcc" ]; then
    echo "  WARNING: 'nvcc' ($(command -v nvcc)) failed a trivial compile, and it's not the known gcc-version-ceiling issue -- CUDA kernel builds below may fail:" >&2
    printf '%s\n' "$out" | tail -5 | sed 's/^/    /' >&2
    return 1
  fi

  echo "  nvcc ($(command -v nvcc)) rejects the default host compiler -- needs gcc <=$max_gcc"

  local conda_bin
  conda_bin="$(command -v conda || true)"
  [ -z "$conda_bin" ] && [ -x /usr/bin/conda ] && conda_bin=/usr/bin/conda
  if [ -z "$conda_bin" ]; then
    echo "  ERROR: no gcc<=$max_gcc found on PATH and no 'conda' available to fetch one." >&2
    echo "  Install an older gcc yourself and export CUDAHOSTCXX to its g++, or switch to a CUDA toolkit that supports the installed gcc." >&2
    return 1
  fi

  # Locate the compiler via `conda run` rather than guessing
  # "<base>/envs/<name>/bin/..." -- conda's actual env storage location
  # (envs_dirs, e.g. ~/.conda/envs) does not have to match its base prefix
  # (e.g. a system-wide /usr base with per-user envs elsewhere).
  local env_name="cuda-gcc${max_gcc}"
  local env_gxx
  env_gxx="$("$conda_bin" run -n "$env_name" command -v x86_64-conda-linux-gnu-g++ 2>/dev/null)"
  if [ -z "$env_gxx" ]; then
    echo "  Creating conda env '$env_name' with gcc $max_gcc as a CUDA-compatible host compiler (one-time, ~1min)..."
    "$conda_bin" create -y -n "$env_name" -c conda-forge "gxx_linux-64=$max_gcc" "gcc_linux-64=$max_gcc" >/dev/null
    env_gxx="$("$conda_bin" run -n "$env_name" command -v x86_64-conda-linux-gnu-g++ 2>/dev/null)"
  fi
  if [ -z "$env_gxx" ] || [ ! -x "$env_gxx" ]; then
    echo "  ERROR: failed to create/find a conda gcc<=$max_gcc environment ('$env_name')." >&2
    return 1
  fi

  export CUDAHOSTCXX="$env_gxx"
  echo "  CUDAHOSTCXX=$CUDAHOSTCXX"

  out="$(_probe_nvcc_compile "$CUDAHOSTCXX")"
  status=$?
  if [ "$status" != 0 ]; then
    echo "  ERROR: still fails to compile even with CUDAHOSTCXX=$CUDAHOSTCXX:" >&2
    printf '%s\n' "$out" | tail -5 | sed 's/^/    /' >&2
    unset CUDAHOSTCXX
    return 1
  fi
}

# Two independent ways a CUDA toolkit can be unusable on one machine (both
# hit while developing this script, on the same box):
#   - nvcc missing from PATH entirely (CUDA installed but not wired into the
#     environment, e.g. no environment-module loaded).
#   - nvcc found, but unusable: some distro CUDA packages ship a toolkit
#     missing static libs their own compiler-ID check hard-requires
#     (surfaces as "cannot find -lcudart_static" from `ld`), and/or the
#     system's default gcc/g++ is newer than that CUDA release supports
#     (see ensure_cuda_host_compiler above). Distros move gcc forward far
#     faster than CUDA toolkits add support for it, so this is common even
#     on a machine where nvcc itself is perfectly findable.
# Try every CUDA install we can find, actually test-compiling a trivial .cu
# with each rather than trusting `command -v nvcc`, applying
# ensure_cuda_host_compiler's conda fallback along the way. Leaves
# CUDA_HOME/CUDACXX/PATH/LD_LIBRARY_PATH/CUDAHOSTCXX exported for whichever
# install (if any) actually works; callers should treat a non-zero return
# as "proceed anyway, CUDA kernel builds may fail" rather than a hard stop.
ensure_working_cuda_toolchain() {
  local cuda_dir cuda_dirs_tried=""
  for cuda_dir in "${CUDA_HOME:-}" "$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || true)")" 2>/dev/null)" /usr/local/cuda /usr/local/cuda-*; do
    [ -n "$cuda_dir" ] && [ -x "$cuda_dir/bin/nvcc" ] || continue
    case " $cuda_dirs_tried " in *" $cuda_dir "*) continue ;; esac
    cuda_dirs_tried="$cuda_dirs_tried $cuda_dir"

    export CUDA_HOME="$cuda_dir"
    export CUDACXX="$cuda_dir/bin/nvcc"
    export PATH="$cuda_dir/bin:$PATH"
    export LD_LIBRARY_PATH="$cuda_dir/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    unset CUDAHOSTCXX

    echo "  Trying $cuda_dir ($("$CUDACXX" --version | tail -1))"
    if ensure_cuda_host_compiler; then
      echo "  -> working"
      return 0
    fi
    echo "  -> not usable, trying the next CUDA install if any"
  done
  echo "  WARNING: no working CUDA toolkit found -- CUDA kernel builds may fail." >&2
  echo "  If one is installed somewhere nonstandard, export CUDA_HOME before running this script." >&2
  return 1
}
