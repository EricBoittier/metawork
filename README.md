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

- **[ASE](https://wiki.fysik.dtu.dk/ase/)** ([repo](https://gitlab.com/ase/ase)) --
  the Atomic Simulation Environment. The lightest-weight way to run a
  metatomic model: see the
  [ASE integration docs](https://docs.metatensor.org/metatomic/latest/engines/ase.html)
  and [`2-running-ase-md.py`](https://github.com/metatensor/metatomic/blob/main/python/examples/2-running-ase-md.py).
- **[TorchSim](https://github.com/TorchSim/torch-sim)** -- batched,
  GPU-native MD in pure PyTorch (no C++/Rust build required). See the
  [TorchSim integration docs](https://docs.metatensor.org/metatomic/latest/engines/torchsim.html)
  and the `5-torchsim-getting-started` / `6-torchsim-batched` examples below.
- **[LAMMPS](https://www.lammps.org/)** ([docs](https://docs.lammps.org/)) --
  via the metatomic-enabled fork at
  [metatensor/lammps](https://github.com/metatensor/lammps). See the
  [LAMMPS integration docs](https://docs.metatensor.org/metatomic/latest/engines/lammps.html).
- **[GROMACS](https://www.gromacs.org/)** ([manual](https://manual.gromacs.org/)) --
  via the metatomic staging branch at
  [metatensor/gromacs](https://github.com/metatensor/gromacs). See the
  [GROMACS integration docs](https://docs.metatensor.org/metatomic/latest/engines/gromacs.html).

Also supported (not currently cloned in this workspace, add them to
`INSTALL_REPOS`/`CLONE_ONLY_REPOS` in `etc/setup-metawork.sh` if you need
them): [i-PI](https://docs.metatensor.org/metatomic/latest/engines/ipi.html),
[PLUMED](https://docs.metatensor.org/metatomic/latest/engines/plumed.html),
[EON](https://docs.metatensor.org/metatomic/latest/engines/eon.html), and
[chemiscope](https://docs.metatensor.org/metatomic/latest/engines/chemiscope.html)
for visualization.


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


# Known issues

Building this ecosystem on a fresh GPU machine (e.g. `cosmopc27`) hit four
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
   too old (found version 12020)`. Fix: install a PyTorch build matching
   what the driver actually supports, e.g.
   ```bash
   uv pip install --python .venv/bin/python 'torch>=2.7' --index-url https://download.pytorch.org/whl/cu121
   ```
   `setup-metawork.sh` now reads the driver's max supported CUDA version
   straight out of `nvidia-smi`'s own header and picks a matching wheel
   channel (`cu121`/`cu124`/`cu126`/default) automatically, then verifies
   `torch.cuda.is_available()` after install and warns if it's still
   `False`.