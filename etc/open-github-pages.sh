#!/usr/bin/env bash
# ============================================================================
# open-github-pages.sh
#
# Opens the GitHub Issues and Pull Requests pages for every repo in the
# metatensor ecosystem used here, each in its own Firefox tab -- handy for a
# quick sweep to check for known bugs (e.g. the featomic_torch / nvidia.cudnn
# issue in the README) or to see what's currently in flight upstream.
#
# By default this opens the *upstream* repos (forks generally don't have
# Issues enabled). Pass --fork to open your own fork's Pull Requests instead
# (useful for checking on PRs you've opened upstream).
# ============================================================================
set -euo pipefail

FORK_OWNER="EricBoittier"
UPSTREAM_ORG="metatensor"

# "name:org" -- org empty defaults to $UPSTREAM_ORG. Keep this in sync with
# the INSTALL_REPOS/CLONE_ONLY_REPOS repo list in setup-metawork.sh.
REPOS=(
  "metatensor:"
  "metatomic:"
  "metatrain:"
  "featomic:"
  "gpu-lite:"
  "hpc-docs:"
  "lj-test:"
  "lammps:"
  "gromacs:"
  "i-pi:i-pi"
  "chemiscope:lab-cosmo"
  "eOn:TheochemUI"
  "plumed2:plumed"
)

pages=(issues pulls)
use_fork=0
if [ "${1:-}" = "--fork" ]; then
  use_fork=1
  pages=(pulls)   # forks don't have their own Issues tab by default
fi

if ! command -v firefox >/dev/null; then
  echo "firefox not found on PATH" >&2
  exit 1
fi

for entry in "${REPOS[@]}"; do
  IFS=':' read -r repo org <<< "$entry"
  owner="${org:-$UPSTREAM_ORG}"
  [ "$use_fork" = 1 ] && owner="$FORK_OWNER"
  for page in "${pages[@]}"; do
    firefox --new-tab "https://github.com/$owner/$repo/$page" >/dev/null 2>&1 &
  done
done
