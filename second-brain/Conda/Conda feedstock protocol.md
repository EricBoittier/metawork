---
tags: [conda, feedstock, protocol]
---
# Conda feedstock protocol

Feedstocks: `~/Documents/lammps-metatomic-feedstock`, `~/Documents/plumed-metatomic-feedstock`
(next to metawork, **not** inside it). Package channel: `metatensor` on anaconda.org.

Needs: [[Conda and uv envs]] (miniforge + `conda-smithy` env).

## Anatomy
- `recipe/meta.yaml` — version, `git_rev`, `git_rev_count`, build number, requirements
- `recipe/build.sh` — the actual build
- `recipe/conda_build_config.yaml` — variant matrix (CUDA, kokkos arch, MPI)
- `conda-forge.yml` — CI/smithy config
- `.ci_support/*.yaml`, `.github/workflows/` — **generated** by rerender; don't hand-edit

## Bump the LAMMPS source
1. Get the source fix merged in `metatensor/lammps` first.
2. In `recipe/meta.yaml`:
   - set `git_rev` to the new commit
   - `git_rev_count` += 1 (if `version` unchanged), otherwise reset it and bump `version`
   - reset `build` to 0 when the version changes
3. Rerender (below), review, commit on a branch, push to fork, open PR.

PR order when several pieces change:
1. engine feedstock others depend on (e.g. plumed-metatomic)
2. `metatensor/lammps` source PR
3. bump `git_rev` in lammps-metatomic-feedstock

## Rerender
**Never** run plain `conda-smithy rerender` on lammps-metatomic — it explodes ~360k variants and hangs at 100% CPU. Use the wrapper:
```bash
cd ~/Documents/metawork
bash etc/rerender-feedstock.sh ~/Documents/lammps-metatomic-feedstock
bash etc/rerender-feedstock.sh ~/Documents/plumed-metatomic-feedstock
```
Then review — don't commit `README.md`:
```bash
cd ~/Documents/lammps-metatomic-feedstock
git status && git diff
git restore --staged README.md && git restore README.md
```
`No changes made. This feedstock is up-to-date.` = nothing to do. Warnings about `Free Disk Space` / Azure token are normal.

## Build locally
```bash
cd ~/Documents/lammps-metatomic-feedstock
ls .ci_support/                                   # pick a variant config
python build-locally.py <config-name-without-.yaml>   # runs in docker on linux
```
Outputs land in `build_artifacts/`. Test-install:
```bash
conda create -n test -c ./build_artifacts -c metatensor -c conda-forge lammps-metatomic
```

## Install for users
```bash
conda config --add channels metatensor
conda config --set channel_priority strict
conda install lammps-metatomic
```

## Gotchas
- Build string encodes `cpu`/`cudaXY_kokkos_arch_*` and MPI flavour; higher build numbers are given to preferred variants (CPU > GPU, openmpi > mpich > nompi).
- `libfabric1 <2.5.1` pin for mpich CPU builds (glibc 2.17).
- Changing a migration/pin (e.g. libtorch) → maintainers expect a rerender in the PR.
