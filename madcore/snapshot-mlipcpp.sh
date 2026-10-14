#!/usr/bin/env bash
# Freeze the mlip.cpp working tree for a job: jobs build exactly the source
# they were submitted with, never a tree that is being edited. Prints the path.
set -euo pipefail
src=/home/boittier/metawork/mlip.cpp
d=/work/cosmo/boittier/lyra/mlipcpp/src-snapshots/$(date +%Y%m%d-%H%M%S)
mkdir -p "$d"
rsync -a --exclude 'build*' --exclude '.git' --exclude 'website' --exclude 'packages' \
    --exclude '__pycache__' "$src/" "$d/"
git -C "$src" rev-parse HEAD > "$d/HEAD"
git -C "$src" diff > "$d/DIFF.patch"
git -C "$src" status --short > "$d/STATUS"
echo "$d"
