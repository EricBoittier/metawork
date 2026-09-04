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

REPOS=(
  metatensor
  metatomic
  metatrain
  featomic
  gpu-lite
  hpc-docs
  lj-test
)

owner="$UPSTREAM_ORG"
pages=(issues pulls)
if [ "${1:-}" = "--fork" ]; then
  owner="$FORK_OWNER"
  pages=(pulls)   # forks don't have their own Issues tab by default
fi

if ! command -v firefox >/dev/null; then
  echo "firefox not found on PATH" >&2
  exit 1
fi

for repo in "${REPOS[@]}"; do
  for page in "${pages[@]}"; do
    firefox --new-tab "https://github.com/$owner/$repo/$page" >/dev/null 2>&1 &
  done
done
