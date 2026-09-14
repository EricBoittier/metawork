# Liquid water dataset → metatensor `DiskDataset`

## Source

Raw data: `deepmodeling/deepmd-kit`, `examples/water/data/{data_0,data_1,data_2,data_3}`
(fetched from GitHub, `master` branch, LGPL-3.0 licensed repository).

This is the well-known 64-molecule bulk liquid water benchmark dataset used in the
original DeePMD / DeepPot-SE papers (Zhang, Han, Wang, Car & E). Each of the 4 shards
(`data_0`..`data_3`) contains 80 independent snapshots of a 192-atom (64 H2O), cubic,
periodic simulation cell, with:

- `box.npy`: (80, 9) cell vectors, Angstrom
- `coord.npy`: (80, 576) atomic positions, Angstrom
- `energy.npy`: (80,) total DFT (PBE) energy, eV
- `force.npy`: (80, 576) atomic forces, eV/Angstrom
- `type.raw` / `type_map.raw`: atom types (0=O, 1=H)

Total: **320 frames**, 192 atoms each.

## Pipeline

1. `scripts/01_build_xyz.py` — reads the raw `.npy`/`.raw` files and writes
   `water_deepmd.xyz`, an ASE-readable extended-XYZ file with `energy` (per-structure,
   in `atoms.info`) and `forces` (per-atom, in `atoms.arrays`). This is metatrain's
   "ASE-readable" dataset format, suitable directly as `read_from: water_deepmd.xyz`
   in a training config.
2. `scripts/02_build_disk_dataset.py` — converts the same data into metatrain's native
   **metatensor `DiskDataset`** format (`water_deepmd.zip`): a zip archive with one
   folder per structure (`0/`, `1/`, ... `319/`), each containing:
   - `system.mta` — the atomic structure as a serialized
     `metatomic.torch.System` (species, positions, cell, PBC)
   - `energy.mts` — a `metatensor.torch.TensorMap` holding the total energy as the
     block value, with a `"positions"` gradient block holding `-forces` (the
     metatensor convention: forces = -dE/dr, so the gradient stores the negated
     force array).
   Built with `metatrain.utils.data.writers.DiskDatasetWriter`, following the
   pattern in `metatrain/examples/0-beginner/01-data_preparation.py`.
3. `scripts/03_verify_dataset.py` — round-trip check: reloads `water_deepmd.zip`
   with metatrain's own `DiskDataset` reader and confirms energies/forces/positions
   match the source `.npy` arrays.

## Units

Angstrom (positions/cell), eV (energy), eV/Angstrom (forces) — deepmd-kit's fixed
unit convention, carried through unchanged.

## Environment used

`/home/boittier/metawork/.venv-upet` (ase 3.29.0, metatensor 0.2.4, metatomic,
metatrain 2026.4, torch 2.13.0).

## Second dataset: `cace_water.zip`

A larger dataset of the *same* physical system (64 H2O = 192 atoms, cubic
periodic box, ~12.42 Å edge) was found already present on this shared filesystem
at `/home/rumiants/pkgs/cace-lr-fit/fit-water/water.xyz` — 1,593 frames (vs. 320
above), with real per-frame energy and per-atom forces, actively used by a
colleague's local CACE (github.com/BingqingCheng/cace) potential-fitting scripts
in the same directory.

- `scripts/04_build_disk_dataset_cace_water.py` reads that file (read-only;
  nothing under `/home/rumiants` is modified) and writes `cace_water.zip`, a
  metatensor `DiskDataset` built the same way as `water_deepmd.zip`.
- `scripts/05_verify_cace_water.py` confirms an exact round-trip against the
  source xyz (all deltas `0.000e+00`).
- **License note**: the `cace-lr-fit` repository's `README.md` states
  `CC BY-NC 4.0` (non-commercial) for that project. That restriction carries
  over to `cace_water.zip` — check before any commercial use. Provenance of the
  underlying DFT calculations is not explicitly documented in the local copy
  (only that it's used as CACE water training data); likely derived from the
  CACE project's own water example/lineage, but unconfirmed.
- `water_deepmd.zip` (from the public, LGPL-3.0 DeePMD-kit source) remains the
  dataset with clean, fully-documented provenance if that matters more than
  frame count.
