#!/usr/bin/env bash
# Fool-proof conda-smithy rerender for metatensor feedstocks.
# Activates ~/miniforge3 env conda-smithy, collapses unused pins (so
# lammps-metatomic does not hang), restores README.md, does not commit or push.
set -euo pipefail

MINIFORGE="${MINIFORGE:-$HOME/miniforge3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/feedstock-rerender/rerender_feedstock.py"

if [[ ! -f "$MINIFORGE/etc/profile.d/conda.sh" ]]; then
  echo "error: miniforge not found at $MINIFORGE" >&2
  echo "install with:" >&2
  echo "  curl -fsSL -o /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-\$(uname)-\$(uname -m).sh" >&2
  echo "  bash /tmp/Miniforge3.sh -b -p \$HOME/miniforge3" >&2
  echo "  \$HOME/miniforge3/bin/conda create -n conda-smithy -c conda-forge conda-smithy -y" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$MINIFORGE/etc/profile.d/conda.sh"
if ! conda env list | grep -q '^conda-smithy '; then
  echo "error: conda env 'conda-smithy' missing; create it with:" >&2
  echo "  conda create -n conda-smithy -c conda-forge conda-smithy -y" >&2
  exit 1
fi
conda activate conda-smithy

exec python "$SCRIPT" "$@"
