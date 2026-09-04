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

ASE
TorchSim
LAMMPS
GROMACS

# Known issues

Building this ecosystem on a fresh GPU machine (e.g. `cosmopc27`) hit three
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