# openmm-ml + metatomic_ase.MetatomicCalculator

Checks that this pattern works end to end:

```python
from metatomic_ase import MetatomicCalculator
from openmmml import MLPotential

calculator = MetatomicCalculator(
    "exported-model.pt",
    device="cuda",
    do_gradients_with_energy=True,
)

potential = MLPotential("ase")
system = potential.createSystem(
    topology,
    calculator=calculator,
)
```

`MLPotential("ase")` resolves to `ASEPotentialImplFactory` (registered via the
`openmmml.potentials` entry point in
[`openmm-ml/setup.py`](../../openmm-ml/setup.py)), whose `addForces` builds an
`ase.Atoms` from the topology and attaches the given calculator
([`openmm-ml/openmmml/models/asepotential.py`](../../openmm-ml/openmmml/models/asepotential.py)).
`MetatomicCalculator`'s constructor takes `model` positionally and everything
else (`device`, `do_gradients_with_energy`, ...) keyword-only
([`metatomic/python/metatomic_ase/src/metatomic_ase/_calculator.py`](../../metatomic/python/metatomic_ase/src/metatomic_ase/_calculator.py)).

Neither `openmm` nor `openmm-ml` were in this workspace's `.venv`/`.venv-upet`
— [`setup_env.sh`](setup_env.sh) builds a throwaway venv with them, reusing
whatever `ase`/`torch`/`metatomic`/`metatomic_ase` is already on the system.

## Reproduce

```bash
cd etc/openmm-metatomic-ase
./setup_env.sh
```

**Toy model** (no network needed, uses the CPU-only checkpoint already in
`metatomic/examples/ase/`) — single-point energy/forces, then a real 200-step
NVE trajectory (checks the calculator survives repeated calls, not just one
`getState()`):

```bash
.venv/bin/python run_toy_model.py cpu
.venv/bin/python run_md.py cpu
```

**`device="cuda"`**, since that checkpoint declares `supported_devices: [cpu]`
only — export a throwaway CUDA-capable copy of the same toy model first:

```bash
.venv/bin/python export_toy_cuda_model.py exported-model-cuda.pt
.venv/bin/python run_toy_model.py cuda exported-model-cuda.pt
.venv/bin/python run_md.py cuda exported-model-cuda.pt
```

**Real model from Hugging Face** — PET-MAD (`lab-cosmo/pet-mad`) is published
as a metatrain checkpoint (`.ckpt`), not an exported `.pt`. `mtt export`
downloads and converts it. This workspace's vendored `metatrain` can't export
that particular checkpoint (see the note below) — swap in the released PyPI
package first:

```bash
.venv/bin/pip uninstall -y metatrain && .venv/bin/pip install --no-deps metatrain
.venv/bin/mtt export lab-cosmo/pet-mad models/pet-mad-dev.ckpt -o exported-petmad.pt
.venv/bin/python run_petmad_water.py exported-petmad.pt cpu
.venv/bin/python run_petmad_water.py exported-petmad.pt cuda
.venv/bin/python run_petmad_md.py exported-petmad.pt cpu
.venv/bin/python run_petmad_md.py exported-petmad.pt cuda
```

Single-point CPU and CUDA energies/forces matched to float32 precision, and
200-step NVE runs (`run_md.py`, `run_petmad_md.py`) conserved total energy to
within ~0.15 kJ/mol std on both the toy model and PET-MAD, on a 4060 Ti + 2070
machine (last checked 2026-09-16).

> **This workspace's vendored `metatrain` can't export that checkpoint right
> now.** `PET._add_output` in
> [`metatrain/src/metatrain/pet/model.py`](../../metatrain/src/metatrain/pet/model.py#L1092)
> treats any rank-1 Cartesian target with a single components axis as
> dipole-like and collapses it to a scalar "charge" head (from
> `5dc68ffd`, "Fix PET's rank-1 Cartesian (dipole) head", 2026-09-07). A
> per-atom `non_conservative_forces` output has exactly that shape too, so
> it gets misclassified the same way, and loading any checkpoint with that
> head (e.g. `pet-mad-dev.ckpt`) then fails with a `state_dict` shape
> mismatch (`[3, 128]` in the checkpoint vs. `[1, 128]` in the freshly-built
> model). The official `metatrain==2026.4` from PyPI does not have this
> regression and exports the same checkpoint cleanly — that's what
> `setup_env.sh` does *not* install by default (it uses the local editable
> checkout instead), so swap it in with `pip install metatrain` inside
> `.venv` if you need this path.

## Gotchas

- `topology.atoms()` must all have `.element` set — `addForces` raises
  otherwise.
- A model's `ModelCapabilities.supported_devices` must actually list `"cuda"`
  for `device="cuda"` to work; `pick_device` raises if it doesn't.
- If a custom model registers `equilibrium_positions` (or similar) as a plain
  tensor attribute rather than via `register_buffer`, `.to(device)` won't move
  it and you'll get a device-mismatch error at `forward()` time — see
  `export_toy_cuda_model.py` vs. the original `metatomic/examples/ase/1-md.py`
  tutorial model.
- `exported-model-cuda.pt` / `exported-petmad.pt` are gitignored scratch
  outputs of the scripts above, not checked in.
- `do_gradients_with_energy=True` also computes stress whenever energy or
  forces are requested. On a non-periodic system that divides by
  `atoms.cell.volume == 0`, so you'll see a harmless
  `RuntimeWarning: invalid value encountered in scalar add` from
  `_full_3x3_to_voigt_6_stress` on every step; the resulting NaN stress is
  just never read.
- `pip install metatrain` won't actually swap in the PyPI release if the
  local editable one is already installed satisfying that requirement —
  `pip uninstall -y metatrain` first (see the PET-MAD command above).
