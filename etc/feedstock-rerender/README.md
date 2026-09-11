# Rerender metatensor conda feedstocks

`conda-smithy rerender` regenerates `.ci_support/*.yaml` and CI workflows from
the recipe + conda-forge pinning. Maintainers expect that on PRs that change
migrations (for example a libtorch pin).

Do **not** run plain `conda-smithy rerender` on `lammps-metatomic-feedstock`:
conda-build explodes unused pins (Python, ROOT, Arrow, …) × kokkos × MPI
(~360k variants) and hangs at 100% CPU after

```text
WARNING: Number of parsed outputs does not match detected raw metadata blocks.
```

Those other warnings (`Free Disk Space` / `Store Build Artifacts` deprecated,
Azure token) are normal. `No changes made. This feedstock is up-to-date.` means
the generated files already match the current smithy + pinning.

## One command

From this metawork checkout (conda is **not** on the default `PATH` on
`cosmopc7`; the wrapper sources `~/miniforge3`):

```bash
bash etc/rerender-feedstock.sh ~/Documents/lammps-metatomic-feedstock
# or plumed:
bash etc/rerender-feedstock.sh ~/Documents/plumed-metatomic-feedstock
```

Then review; do not commit `README.md`; do not push until you have looked at
the diff:

```bash
git -C ~/Documents/lammps-metatomic-feedstock status
git -C ~/Documents/lammps-metatomic-feedstock diff
```

If smithy staged a commit message, drop README first:

```bash
cd ~/Documents/lammps-metatomic-feedstock
git restore --staged README.md
git restore README.md
```

## Which clone

Keep feedstock checkouts **next to** metawork, not inside it:

| use this | not this |
| --- | --- |
| `~/Documents/lammps-metatomic-feedstock` | `~/Documents/metawork/lammps-metatomic-feedstock` |
| `~/Documents/plumed-metatomic-feedstock` | `~/Documents/metawork/plumed-metatomic-feedstock` |

A nested clone inside metawork is a separate git repo. `git add -A` in
metawork will try to record it as a gitlink. The wrapper refuses those
paths unless you pass `--force`.

## Env

- Miniforge: `~/miniforge3` (not on default `PATH`)
- Env: `conda-smithy` (`conda create -n conda-smithy -c conda-forge conda-smithy`)
- Tool: `conda-smithy` (update with `conda update -n conda-smithy conda-smithy`)

`etc/rerender-feedstock.sh` activates that env, patches `explode_variants` so
unused pins collapse to one value, restores `README.md`, and does not commit
or push.

## After a LAMMPS source fix

The lammps feedstock clones `https://github.com/metatensor/lammps.git` at
`recipe/meta.yaml` `git_rev`. A compile fix (for example a missing
`#include <memory>`) has to merge there first; then bump `git_rev` and
`git_rev_count` in the feedstock and rerender/rebuild. Example pull order:

1. engine feedstock that others depend on (e.g. plumed-metatomic)
2. `metatensor/lammps` source PR
3. bump `git_rev` in lammps-metatomic-feedstock, then merge that PR
