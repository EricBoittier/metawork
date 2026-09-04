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
    echo "h5py is not installed in the venv. Install it with:" >&2
    echo "  uv pip install --python $VPY h5py" >&2
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
