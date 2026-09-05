# metawork

Repo to install the entire metatensor ecosystem for testing!

```bash
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
bash ~/metawork/etc/setup-metawork.sh
```


# Programs

Simulation engines with a metatensor/metatomic integration, so a model built
with this stack can be dropped in as the potential:

Installed straight into the shared venv by `setup-metawork.sh` (pure/mostly
Python, no special native build):

- **[ASE](https://wiki.fysik.dtu.dk/ase/)** ([repo](https://gitlab.com/ase/ase)) --
  the Atomic Simulation Environment, pulled in as a dependency. The
  lightest-weight way to run a metatomic model: see the
  [ASE integration docs](https://docs.metatensor.org/metatomic/latest/engines/ase.html)
  and [`2-running-ase-md.py`](https://github.com/metatensor/metatomic/blob/main/python/examples/2-running-ase-md.py).
- **[TorchSim](https://github.com/TorchSim/torch-sim)** -- batched,
  GPU-native MD in pure PyTorch. Would install via metatomic's `torchsim`
  extra (no separate repo needed), but that extra is currently broken by an
  upstream dependency conflict -- see Known Issues below. See the
  [TorchSim integration docs](https://docs.metatensor.org/metatomic/latest/engines/torchsim.html)
  and the `5-torchsim-getting-started` / `6-torchsim-batched` examples below.
- **[i-PI](https://ipi-code.org/)** ([repo](https://github.com/i-pi/i-pi),
  imports as `ipi`) -- a universal, Python-based force engine; metatomic
  support is in the official version. See the
  [i-PI integration docs](https://docs.metatensor.org/metatomic/latest/engines/ipi.html)
  and [hpc-docs' i-PI notes](https://github.com/metatensor/hpc-docs/blob/main/CSCS-Alps/molecular-dynamics-with-i-Pi.md).
- **[chemiscope](https://chemiscope.org/)** ([repo](https://github.com/lab-cosmo/chemiscope)) --
  interactive structure/property viewer. Its build bundles JS assets via
  `npm` and needs **node >=20**; `setup-metawork.sh` checks for this and
  skips chemiscope with a warning instead of failing the whole run if it's
  missing/too old (this was the case on `cosmopc7`, which ships node/npm
  too old to build it).

Cloned for reference but *not* auto-built (each has its own large,
non-Python/CMake+MPI-style build, too machine-specific to script here --
follow the linked docs for the actual build):

- **[LAMMPS](https://www.lammps.org/)** ([docs](https://docs.lammps.org/)) --
  via the metatomic-enabled fork at
  [metatensor/lammps](https://github.com/metatensor/lammps). See the
  [LAMMPS integration docs](https://docs.metatensor.org/metatomic/latest/engines/lammps.html).
- **[GROMACS](https://www.gromacs.org/)** ([manual](https://manual.gromacs.org/)) --
  via the metatomic staging branch at
  [metatensor/gromacs](https://github.com/metatensor/gromacs). See the
  [GROMACS integration docs](https://docs.metatensor.org/metatomic/latest/engines/gromacs.html).
- **[eOn](https://eondocs.org/)** ([repo](https://github.com/TheochemUI/eOn)) --
  long-timescale transition-state/kinetics engine; metatomic support is in
  the official version. See the
  [eOn integration docs](https://docs.metatensor.org/metatomic/latest/engines/eon.html).
- **[PLUMED](https://www.plumed.org/)** ([repo](https://github.com/plumed/plumed2))
  -- enhanced-sampling/free-energy library; metatomic support is in the
  official development version. See the
  [PLUMED integration docs](https://docs.metatensor.org/metatomic/latest/engines/plumed.html).


# Models

**[upet](https://github.com/lab-cosmo/upet)** (fork:
[EricBoittier/upet](https://github.com/EricBoittier/upet)) -- the lab's
universal interatomic potentials (PET-MAD, PET-OAM, PET-MAD-DOS), successor
to the now-deprecated PET-MAD repo. [Docs](https://lab-cosmo.github.io/upet/latest/).

```py
from upet.calculator import UPETCalculator
calculator = UPETCalculator(model="pet-mad-s", version="1.5.0", device="cuda")
```

upet pins `metatrain>=2026.4,<2026.5` (a released version), which the
editable dev `metatrain` checkout in the main venv above does not satisfy --
installing it there would silently swap that editable checkout for the
pinned PyPI release, breaking "edit metatrain, see it everywhere" for the
rest of the ecosystem. So `setup-metawork.sh` gives upet **its own venv**
(`.venv-upet`) instead, with its own pinned `metatrain==2026.4` -- the main
venv's editable `metatrain` is left untouched.

```bash
source .venv-upet/bin/activate
# or: uv run --python .venv-upet/bin/python <command>
```


# Reference implementations

**[iris-infra](https://github.com/sirmarcel/iris-infra)** is the JAX PET /
PETLR trainer (`iris.pet.PET`, `iris.pet.PETLR`, `iris-train`).
**[lorem-jax](https://github.com/lab-cosmo/lorem-jax)** is the official JAX
LOREM / `LoremBEC` package. Both are git submodules, not ecosystem pip
installs.

```bash
git submodule update --init
# or re-run setup-metawork.sh, which inits submodules from .gitmodules
```

A side-by-side of iris PETLR, paper LOREM / lorem-jax, metatrain
`experimental.lorem`, and PET `long_range` lives in
[`metatrain/src/metatrain/experimental/lorem/README.md`](metatrain/src/metatrain/experimental/lorem/README.md).


# Documentation

Per-project hosted docs:

- [metatensor](https://docs.metatensor.org/)
- [metatomic](https://docs.metatensor.org/metatomic/)
- [featomic](https://metatensor.github.io/featomic/index.html)
- [metatrain](https://docs.metatensor.org/metatrain/)
- [hpc-docs](https://github.com/metatensor/hpc-docs) -- install/run
  instructions for this ecosystem on HPC clusters, including
  [CSCS Alps](https://github.com/metatensor/hpc-docs/blob/main/CSCS-Alps/metatrain.md)

Open all of them at once in Firefox:
```bash
for url in \
  https://docs.metatensor.org/ \
  https://docs.metatensor.org/metatomic/ \
  https://metatensor.github.io/featomic/index.html \
  https://docs.metatensor.org/metatrain/ \
  https://github.com/metatensor/hpc-docs
do
  firefox --new-tab "$url" &
done
```

To instead open every repo's GitHub Issues + Pull Requests pages (e.g. to
check for known bugs like the `featomic_torch` one below, or to see what's
currently in flight upstream):
```bash
bash etc/open-github-pages.sh          # upstream issues + PRs
bash etc/open-github-pages.sh --fork   # your own fork's PRs
```


# Examples

- **metatensor core** -- [hosted gallery](https://docs.metatensor.org/latest/examples/core/index.html) /
  [source](https://github.com/metatensor/metatensor/tree/main/python/examples/core):
  `TensorMap` basics, sparsity, gradients, DLPack interop.
- **metatensor learn** -- [hosted gallery](https://docs.metatensor.org/latest/examples/learn/index.html) /
  [source](https://github.com/metatensor/metatensor/tree/main/python/examples/learn):
  datasets/dataloaders, equivariant `nn.Module`s.
- **metatomic** -- [source](https://github.com/metatensor/metatomic/tree/main/python/examples):
  exporting an atomistic model, running ASE MD, neighbor lists, profiling,
  and getting started with TorchSim (single + batched).
- **metatrain, beginner** -- [hosted gallery](https://docs.metatensor.org/metatrain/latest/generated_examples/0-beginner/index.html) /
  [source](https://github.com/metatensor/metatrain/tree/main/examples/0-beginner):
  data prep, training from scratch, fine-tuning, parity plots, running the
  result in ASE.
- **metatrain, advanced** -- [hosted gallery](https://docs.metatensor.org/metatrain/latest/generated_examples/1-advanced/index.html) /
  [source](https://github.com/metatensor/metatrain/tree/main/examples/1-advanced):
  transfer learning, LLPR (uncertainty), ZBL, generic targets, FlashMD,
  multi-GPU, DOS training.


# Datasets

The bundled example `.xyz` files above (e.g. `ethanol_reduced_100.xyz`) are
tiny 100-structure teaching samples checked into the repos themselves --
nothing to download. For real training/benchmarking, `etc/download-datasets.sh`
fetches full reference datasets from [sGDML](http://www.sgdml.org/#datasets)
into a data directory (default `~/data`) and converts them to extended-XYZ
(energy in eV, forces in eV/A, read back via the standard
`atoms.get_potential_energy()` / `atoms.get_forces()` ASE calculator
convention -- not `atoms.info`/`.arrays`).

```bash
bash etc/download-datasets.sh                       # ethanol -> ~/data/md17/
bash etc/download-datasets.sh aspirin naphthalene    # multiple molecules
bash etc/download-datasets.sh --data-dir /scratch/data ethanol
```

Available MD17 molecules (the classic 8-molecule benchmark):
`aspirin`, `benzene2017`, `ethanol`, `malonaldehyde`, `naphthalene`,
`salicylic`, `toluene`, `uracil`. The larger MD22 molecules work too, e.g.
`Ac-Ala3-NHMe`, `AT-AT`, `AT-AT-CG-CG`, `buckyball-catcher`, `DHA`,
`double-walled_nanotube`, `stachyose`.

Each molecule downloads as a raw `.npz` (the full trajectory -- hundreds of
thousands of near-duplicate frames, kept as-is) plus a ready-to-train
`.xyz` with a random 1000-frame subsample
(`etc/md17_npz_to_xyz.py --n-samples all` for the full thing instead).
Re-running is idempotent -- already-downloaded `.npz` files are skipped.

QM7-X (small organics, ~4.2 M PBE0+MBD structures, 40+ properties) is a
separate path:

- energy + forces via OpenQDC — [`etc/openqdc_metatomic/`](etc/openqdc_metatomic/)
- original Zenodo HDF5, several endpoints of different tensor character —
  [`etc/qm7x_zenodo/`](etc/qm7x_zenodo/)
  ([record 4288677](https://zenodo.org/records/4288677));
  [`convert/`](etc/qm7x_zenodo/convert/) writes `~/data/qm7x/qm7x.xyz`
  (`h5py` is installed by `setup-metawork.sh`),
  [`train/`](etc/qm7x_zenodo/train/) runs `mtt train` against the YAMLs in
  [`options/`](etc/qm7x_zenodo/options/)


# Known issues

Building this ecosystem on a fresh GPU machine (e.g. `cosmopc27`) hit seven
separate problems, in the order you'll likely see them:

1. **`cargo`/`rustc` present but not working.** On some machines `cargo` is a
   rustup shim with no default toolchain configured, so it prints
   `error: rustup could not choose a version of cargo to run` instead of a
   version -- and `metatensor-core`'s CMake version check chokes on that in a
   confusing way (`CMake Error ... Unknown arguments specified`). Fix:
   `rustup default stable`. `setup-metawork.sh` now detects and runs this
   automatically.

2. **`nvcc` not on `PATH`.** `nvidia-smi` working just means the driver is
   there; building CUDA kernels (`metatensor-torch`, `metatomic-torch`,
   `featomic-torch`) needs the CUDA *toolkit*'s `nvcc` compiler too, which
   isn't always wired into `PATH` by default. Symptom:
   `CMake Error ... No CMAKE_CUDA_COMPILER could be found`. Fix: point at
   wherever the toolkit actually lives, e.g.
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export CUDACXX=/usr/local/cuda/bin/nvcc
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
   ```
   `setup-metawork.sh` now searches `/usr/local/cuda*` for this automatically
   when `nvcc` is missing but a driver is present.

3. **`featomic_torch`'s `setup.py` crashes on CUDA-13 wheels.** It builds a
   `cudnn_root` path with `os.path.dirname(nvidia.cudnn.__file__)`, assuming
   the `nvidia.cudnn` package always has a real `__file__`. With the
   CUDA-13-targeted `torch` wheel (the default one on a machine with a
   working driver), `nvidia.cudnn` comes back as a namespace package instead,
   so `__file__` is `None` and `os.path.dirname(None)` raises a `TypeError`
   that the surrounding `try/except ImportError` doesn't catch. This is an
   upstream bug, not a local environment problem. `setup-metawork.sh` patches
   the file automatically (idempotent -- no-op once upstream fixes it, or on
   a repeat run) to fall through to the existing PyTorch-bundled-CuDNN
   fallback instead of crashing.

4. **PyTorch installs fine but `torch.cuda.is_available()` is `False`.**
   A working driver isn't the same as a driver that can run *any* CUDA
   build -- each driver caps the newest CUDA runtime it supports. On
   `cosmopc27`, driver `535.309.01` tops out at CUDA 12.2, but PyTorch's
   default install pulls the newest bundled CUDA build (`cu130`, i.e. CUDA
   13.0), which that driver can't run. Symptom: everything imports fine, but
   at the first CUDA call you get
   `UserWarning: CUDA initialization: The NVIDIA driver on your system is
   too old (found version 12020)`.

   `setup-metawork.sh` picks a matching wheel (`cu121` on that machine --
   the newest torch there is `2.5.1+cu121`) and **pins** it. Without the
   pin, `uv pip install -e metatensor[torch]` later replaces it with
   `2.13.0+cu130` from PyPI (the version constraint is satisfied either
   way; uv does not look at the CUDA runtime). If a venv is already in
   that state, restore the matching wheel and rebuild the torch extensions
   without a full setup:

   ```bash
   bash etc/fix-torch-cuda.sh
   ```

4b. **Everything installs into the wrong directory, silently.**
   `setup-metawork.sh` used to hardcode `BASE_DIR="$HOME/Documents/metawork"`
   -- a path from investigating this on `cosmopc7`, where the project
   genuinely lives at `~/Documents/metawork`. On `cosmopc27` the project
   lives at `~/metawork` instead (no `Documents/`), so every run silently
   built a second, orphaned tree at `~/Documents/metawork` while
   `download-datasets.sh` (which correctly derives its paths from its own
   script location) kept looking in `~/metawork/.venv` and finding nothing.
   No error at the time -- both trees "worked" independently, just not
   together. Fixed by deriving `BASE_DIR` from the script's own location
   (`etc/../..`) in every script, so this works wherever the project folder
   actually is. If you hit this before the fix: the orphaned tree's
   editable installs point at its own absolute paths, so don't just `mv` it
   into place -- delete it and re-run fresh at the correct location.

5. **chemiscope needs a newer `node`/`npm` than the system default.** Its
   `pip install` runs `npm ci` to bundle the JS widget, which requires
   **node >=20**. On `cosmopc7` the system npm is 9.2.0 (needs node >=20,
   found an older one), so the build fails with
   `npm ERR! notsup Not compatible with your version of node/npm`. This
   would abort the whole script (everything after chemiscope in
   `INSTALL_REPOS` never gets installed) if left uncaught, since one failed
   `uv pip install` under `set -e` stops the run. `setup-metawork.sh` now
   checks the `node` major version during the toolchain check and skips
   chemiscope with a warning instead of failing, so the rest of the
   ecosystem still installs. Fix, if you actually need chemiscope: install a
   newer node without root via [nvm](https://github.com/nvm-sh/nvm)
   (`nvm install 20`), then re-run the script.

6. **`torch.cuda.is_available()` stays `False` even after issue 4's fix,
   or the install becomes unsatisfiable outright.** Two separate bugs, both
   in the `torch` install command, hit back to back:
   - `uv pip install "torch>=2.7"` is a no-op if a `torch` already
     satisfying `>=2.7` is installed (e.g. a default `cu130` build left
     over from a run before this script picked the right channel),
     *regardless* of `--index-url` -- version satisfaction alone doesn't
     check which CUDA build is present, so the driver-matched channel from
     issue 4 silently had no effect on a machine that had already run this
     script once. Fixed by adding `--reinstall-package torch`.
   - Pinning `torch>=2.7` at all was itself wrong: PyTorch's *minimum*
     supported CUDA version keeps rising with each release, so older
     channels stop shipping new versions entirely (`cu121` tops out around
     `torch==2.5.1` -- there is no `torch>=2.7` build on that channel,
     full stop). On `cosmopc27` this showed up as
     `No solution found ... only torch<=2.5.1+cu121 is available and you
     require torch>=2.7`. The `>=2.7` floor came from metatrain's `dpa3`
     extra, which we don't install by default -- it never belonged on the
     shared `torch` install. Fixed by dropping the floor entirely and
     letting uv pick the newest version actually available on the chosen
     channel.

7. **`metatomic[torch,torchsim]` fails to resolve, and it is not just an
   extras-combination problem.** `metatomic-torch` (installed by the
   `torch` extra) depends on `metatomic-ase`, which requires
   `vesin>=0.6.0,<0.7`; `metatomic-torchsim` requires `vesin>=0.5.6,<0.6`
   *and* depends on `metatomic-torch` itself -- so the conflict is baked
   into `metatomic-torchsim`'s own dependency graph at this dev snapshot,
   not something combining extras causes. Confirmed by testing
   `metatomic[torchsim]` completely alone, in its own empty venv with
   nothing else installed: identical unsatisfiable-dependencies error. So
   **a separate venv does not work around this one** -- it's an upstream
   version-pin mismatch that has to be fixed upstream (or worked around
   with a resolver override forcing a single `vesin` version, which is
   fragile and not done here). `setup-metawork.sh` installs `metatomic`
   with just the `torch` extra (no `torchsim`) until upstream syncs those
   pins -- so TorchSim isn't currently usable from this setup at all. Watch
   [metatensor/metatomic](https://github.com/metatensor/metatomic) for the
   fix and re-add `torchsim` to `INSTALL_REPOS` in `setup-metawork.sh` once
   it lands.

8. **`uv pip install -e metatensor[torch]` fails with
   `AttributeError: 'Version' object has no attribute '__replace__'`.**
   metatensor's `setup.py` bumps its own dev version with
   `packaging.version.Version.__replace__(...)`, an API only added in
   `packaging` 26.0. Its `pyproject.toml` only requires `packaging >=23`
   for the build environment though, and `download.pytorch.org/whl/*`
   (added as an `--extra-index-url` so CPU/CUDA torch wheels resolve
   correctly) turns out to mirror a handful of plain PyPI packages torch
   itself depends on -- `packaging` included, capped around 24.x. uv's
   default index-strategy (`first-index`) stops looking for a package the
   moment *any* configured index has it, so it settled on that old, capped
   `packaging` instead of checking PyPI for a newer one, and the build
   failed before ever reaching metatensor's own code. Fixed by adding
   `--index-strategy unsafe-best-match` to `uv_pip_keep_torch` in
   `_torch_cuda.sh`, so uv considers every configured index and picks the
   best version instead of stopping at the first hit.

9. **Immediately after fixing issue 8, `uv pip install -e
   metatomic[torch]` becomes unsatisfiable: "metatomic-torch depends on
   torch==2.13.\* and torch==2.14.0+cpu".** PyPI shipped `torch==2.14.0`
   two days before this was diagnosed. metatomic-torch's (and
   featomic-torch's, metatrain's) own `build-system.requires` pull in the
   latest PyPI release of `metatensor-torch` to build/link against (they
   can't see our local editable checkout during their own isolated build),
   and that release was itself published pinned to `torch==2.13.*` --
   whatever was newest when *it* was built. Meanwhile nothing holds our
   own locally-built `metatensor-torch` back from chasing PyPI's newest
   torch. The two drift apart the moment PyPI ships a torch newer than
   what the last metatensor-torch release was built against, and a plain
   `-c constraints.txt` file doesn't help -- constraints only scope the
   top-level install graph, not each package's own isolated build
   resolution. `--exclude-newer-package torch=<date>` does apply there too
   (it hides newer releases at the index level, for every lookup,
   including ones inside another package's build isolation), so
   `_torch_cuda.sh` now anchors `torch` behind a `TORCH_EXCLUDE_NEWER_DATE`
   cutoff wherever it's installed. This is a manual, dated pin --
   `setup-metawork.sh` will need it bumped (or removed) once upstream
   republishes the `-torch` packages against `torch==2.14`.

10. **Once issues 8 and 9 are fixed, the full `setup-metawork.sh` run
    still ends with `metatensor`/`metatensor-torch` reported as plain PyPI
    releases (`0.2.4`/`0.10.4`), not the local editable checkout, even
    though "Installing metatensor [torch]" clearly builds and installs the
    editable one first.** metatomic-torch, featomic-torch and metatrain
    all declare `metatensor-torch >=0.10.0,<0.11` as an install
    dependency, and the local metatensor checkout's own dev-version
    scheme (bumping the minor version for every commit since the last
    tag) has drifted past that ceiling to `0.11.0.dev...` -- so installing
    any of those three *after* metatensor silently swaps the just-built
    editable `metatensor-torch` back out for the PyPI `0.10.4` release
    that actually satisfies their pin, and each subsequent repo in
    `INSTALL_REPOS` that shares the same ceiling leaves it there. This
    isn't a bug introduced by anything above -- it's a pre-existing
    version-pin mismatch between this metatensor checkout and the last
    metatomic/featomic/metatrain releases synced against it, and it never
    surfaced before because issue 8 blocked the very first install step.
    Left alone, `INSTALL_REPOS`'s fixed order (metatensor, metatomic,
    featomic, metatrain, ...) happens to settle into a stable, working
    state where only the *last* repo touching a given package keeps its
    local editable copy -- on this machine that meant featomic and
    metatrain (the two repos with actual local changes per `git status`)
    stayed editable, while metatensor and metatomic (untouched locally)
    settled on PyPI releases. If you need to edit metatensor itself and
    see that reflected in metatomic/featomic/metatrain too, this needs an
    actual decision (pin the metatensor checkout to a tag/commit before
    the version crossed `0.11`, or patch the downstream repos'
    `metatensor-torch` upper bound the way `_torch_cuda.sh` already patches
    featomic's cudnn bug) -- not done here.