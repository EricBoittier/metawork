# Shared paths for qm7x_zenodo helper scripts. Source from convert/train/inspect.
# QM7X_DIR is this file's directory (etc/qm7x_zenodo).

QM7X_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$QM7X_DIR/../.." && pwd)"
VPY="$REPO_DIR/.venv/bin/python"
MTT="$REPO_DIR/.venv/bin/mtt"
OPTIONS_DIR="$QM7X_DIR/options"
CONVERT_PY="$QM7X_DIR/zenodo_to_metatensor.py"
DEFAULT_DATA_DIR="${QM7X_DATA_DIR:-$HOME/data/qm7x}"

# Zenodo record 4288677: eight HDF5 shards, named by a molecule-id range.
SHARDS=(1000 2000 3000 4000 5000 6000 7000 8000)

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

require_venv() {
  if [ ! -x "$VPY" ]; then
    echo "No venv python at $VPY -- run etc/setup-metawork.sh first." >&2
    exit 1
  fi
}

require_h5py() {
  require_venv
  if ! "$VPY" -c "import h5py" >/dev/null 2>&1; then
    echo "h5py is not installed in the venv (needed to read QM7-X HDF5)." >&2
    echo "  bash $REPO_DIR/etc/setup-metawork.sh          # installs it" >&2
    echo "  uv pip install --python $VPY h5py             # or just this" >&2
    exit 1
  fi
}

require_mtt() {
  require_venv
  if [ ! -x "$MTT" ]; then
    echo "No mtt at $MTT -- is metatrain installed in the venv?" >&2
    exit 1
  fi
}

# Training YAMLs read qm7x.xyz (and polarizability_spherical.mts / qm7x.zip)
# relative to the working directory. Point a run folder at a data dir.
link_train_inputs() {
  local data_dir="$1"
  local run_dir="$2"
  mkdir -p "$run_dir"
  for name in qm7x.xyz polarizability_spherical.mts qm7x.zip; do
    if [ -e "$data_dir/$name" ]; then
      ln -sfn "$data_dir/$name" "$run_dir/$name"
    fi
  done
}

# Small Zenodo sidecars (HDF5 key list, duplicate-molecule ids). Best-effort.
fetch_zenodo_info() {
  local cache="$1"
  mkdir -p "$cache"
  "$VPY" - "$cache" <<'PY'
import sys
import urllib.request
from pathlib import Path

cache = Path(sys.argv[1])
base = "https://zenodo.org/records/4288677/files"
for name in ("README.txt", "DupMols.dat"):
    dest = cache / name
    if dest.exists():
        continue
    url = f"{base}/{name}?download=1"
    req = urllib.request.Request(url, headers={"User-Agent": "metawork-qm7x"})
    try:
        with urllib.request.urlopen(req, timeout=60) as src:
            dest.write_bytes(src.read())
        print(f"  fetched {name} -> {dest}")
    except Exception as exc:
        print(f"  skip {name}: {exc}")
PY
}
