# Metatomic notebook lab

A local course in the emerging **core Python API**: 22 executable marimo notebooks,
covering structures, energies, forces, stress, simulation, and model/engine design.
The project belongs to **metawork**, beside `metatomic/`. It is independent teaching
material for the development API, following the published metatomic examples.

## Open the collection

```sh
cd /path/to/metawork/metatomic-marimo
uv sync --python 3.12
uv run python -m metatomic_marimo.lab
# open http://localhost:2719
```

The lab has grouped lessons, topic search, source links and live notebook views.
Switching between visited lessons preserves their controls. Use **Open full page**
for a wider reading view; use **Source** for the complete Python file.
To edit a lesson or open marimo's native directory workspace:

```sh
uv run marimo edit notebooks/03_jax_energy_forces_stress.py
uv run marimo edit notebooks/ --port 2722
```

Every computational cell displays its executed source above the result. Copy the
calculation and its imports; replace slider `.value` references with your chosen
values. `mo.show_code`, `mo.md` and layout calls are presentation wrappers.
The `metatomic_marimo` helpers are local cookbook code; their names are not additions
to the public `metatomic` namespace.

## Choose a study path

- **First calculation:** 00 → 01 → 02, then 03 for JAX or 04 for Torch. All three
  backends reproduce the same harmonic energy, force and stress.
- **From model to engine:** 06 → 07 → 05 → 08 → 09. Follow pair topology, labeled
  outputs, relaxation, ASE dynamics and synchronized profiling.
- **Native reference and API design:** 18 → 19 → 20 → 21 → 10. Execute the accepted
  C++ LJ plugin, derive two Python kernels and use parity to inform the missing APIs.
- **Integration work:** choose 11–17 after 02, 07 and 18. Each design note contains
  a runnable boundary fixture, a migration specification and a worked exercise.
  These fixtures do not execute the proposed production engine integrations.

## Lesson catalog

### Core tutorials

| Lesson | Calculation or lesson outcome |
| --- | --- |
| [00 — Meet System](notebooks/00_system.py) | Build a structure with NumPy, JAX, or Torch. Explore positions, atomic types, and the periodic cell. |
| [01 — Metadata & pair lists](notebooks/01_metadata_and_pairs.py) | Describe quantities and capabilities, attach pair lists, and inspect custom data. |
| [02 — NumPy model](notebooks/02_numpy_model.py) | Explore a harmonic pair potential with analytic energy, forces, and stress. |
| [03 — JAX derivatives](notebooks/03_jax_energy_forces_stress.py) | Differentiate a System pytree. Compare force and stress formulas with explicit strain derivatives. |
| [04 — Torch autograd](notebooks/04_torch_energy_forces_stress.py) | Start an autograd graph from System arrays and compare eager and compiled tensor kernels. |
| [05 — Relaxation engine](notebooks/05_numpy_engine.py) | Follow fixed-cell relaxation through a linked energy map, structures, and force arrows. |
| [06 — Differentiable pair lists](notebooks/06_neighbor_lists.py) | Build a periodic ASE neighbor list, attach core pair data, and differentiate a pair kernel. |
| [07 — Outputs & model contract](notebooks/07_outputs_and_model_contract.py) | Label energy gradients, serialize predictions, and inspect which model APIs are available. |
| [08 — ASE dynamics · 3 backends](notebooks/08_ase_dynamics.py) | Write an ASE calculator around core System and run NVE dynamics with NumPy, JAX, or Torch. |
| [09 — Batching & profiling](notebooks/09_batching_and_profiling.py) | Batch JAX predictions, check Torch parity, time synchronized calls, and inspect profiler events. |

### Reference models

| Lesson | Calculation or lesson outcome |
| --- | --- |
| [18 — Native LJ: load and execute](notebooks/18_lj_c.py) | Inspect the accepted C++ plugin, satisfy pair requests, and turn energy gradients into forces. |
| [19 — NumPy LJ: analytic derivatives](notebooks/19_lj_numpy.py) | Write an energy-shifted LJ kernel, check derivatives, and compare labeled native outputs. |
| [20 — Torch LJ: eager and compiled](notebooks/20_lj_torch.py) | Author an nn.Module, verify compiled values and gradients, then check native parity. |
| [21 — C++ LJ contract & parity](notebooks/21_cpp_lj_contract.py) | Execute the upstream C++ LJ plugin and compare NumPy, Torch and JAX outputs, gradients and selection semantics. |

### Integration design notes

| Lesson | Calculation or lesson outcome |
| --- | --- |
| [11 — LAMMPS: forces and virial](notebooks/11_lammps.py) | Convert configurational stress to virial; specify pair, fix, compute and MPI responsibilities. |
| [12 — GROMACS: unit conversion](notebooks/12_gromacs.py) | Derive energy, force and stress conversion factors; verify virtual-work invariance. |
| [13 — PLUMED: bias force chain rule](notebooks/13_plumed.py) | Differentiate a distance restraint, then check the bias forces against finite differences. |
| [14 — ASE: calculator state](notebooks/14_ase_plan.py) | Exercise cached results and geometry invalidation; specify the core calculator boundary. |
| [15 — i-PI: atomic units and virial](notebooks/15_ipi.py) | Convert energy, forces and virial to atomic units and verify the round trip. |
| [16 — Chemiscope: property targets](notebooks/16_chemiscope_plan.py) | Build structure and atom properties with units, force arrows and copied geometry. |
| [17 — TorchSim: packed geometry](notebooks/17_torchsim.py) | Reduce per-atom energies by system and expose row/column cell conventions with a skew cell. |

### API design

| Lesson | Calculation or lesson outcome |
| --- | --- |
| [10 — API progress & prototypes](notebooks/10_api_progress_and_prototypes.py) | Track every Python API todo, exercise errors and JSON, and explore model, loader, and engine prototypes. |

## Scientific conventions

Geometry uses row-vector cells and angstrom; energies are eV, forces eV/angstrom,
and configurational stress eV/angstrom³. Stress is positive in tension; pressure
is `-trace(stress)/3`. Homogeneous strain deforms **both** positions and cell.
The core `metatomic.System` differs from the established `metatomic.torch.System`.
All examples run on CPU in float64, with small analytic potentials and no downloaded
weights or datasets.

The harmonic chapters use two atoms in a fully periodic orthogonal cell. They are
not general lattice sums or triclinic neighbor searches. Derivatives assume a
nonsingular cell, nonzero pair separation and no image switch. Chapter 06 uses a
**shifted-force** LJ potential for a smooth cutoff; chapters 18–21 deliberately use
the accepted native model's **energy-shifted** LJ potential. These potentials have
different forces near cutoff and must not be used as interchangeable oracles.

JAX cannot yet transform a core System with attached pair lists/custom data. Chapter
06 keeps topology fixed and reconstructs vectors inside the kernel. An engine must
rebuild topology when needed. Torch DLPack shares storage without preserving autograd
history; start fresh leaves after extracting geometry. `aot_eager` demonstrates graph
capture and derivative correctness, not optimized compiler performance.

## Structure and property views

Geometry lessons include chemiscope 3D atoms, cell vectors and force arrows, with
an ASE static xy projection as a fallback. Model results include coordinate/force
tables, all nine stress components and pressure. Energies and stresses are explicitly
structure properties; the course does not invent per-atom partitions.

Red force arrows use 0.25 angstrom per (eV/angstrom); this changes display geometry
only. The numerical values retain their physical units. Relaxation and dynamics
include linked trajectory maps; dynamics also records kinetic/total energy and
stress. Expand **Coordinates, forces, cell, and stress** or **ASE static plot**
for a numerical or WebGL-free view. Display adapters copy arrays outside autodiff.
HTML exports preserve saved results; they cannot rerun Python controls.

## API status and design boundary

`System`, metadata classes and JSON serialization are available on the pinned
branch. Public `Model`, `NumpyModel`, `TorchModel`, `JaxModel` and `load_model` remain
absent. Chapter 10's subclasses/manifest are explicitly local design prototypes.
Chapters 18–21 execute the actual C++ plugin through an experimental private Python
bridge; that does not complete a public native-execution or plugin API.

The native LJ reference supports explicit position gradients, but does not advertise
strain gradients. No native stress parity is claimed. Checked system-energy execution
currently rejects selections omitting an entire input system; the tests record this
limitation. See [API status](ROADMAP.md) for every requested work item and branch.

## Reproducible environment

Prerequisites: `uv`, Python 3.12, a C/C++ toolchain and Rust/Cargo >= 1.88. The isolated
`.venv` and `uv.lock` record Python dependencies. `metatomic-core` is a noneditable
local dependency from `../metatomic/python/metatomic_core`, built at
**b02b9ff3df6458cca21fa8690d229bbda97e6f78** (`python-wrap-system`, PR 325).
The sibling Git revision is not pinned by the lockfile. Check it separately:

```sh
git -C ../metatomic rev-parse HEAD
# After an intentional source change, rebuild:
uv sync --reinstall-package metatomic-core
```

The accepted C++ LJ source at `12b24b14` is an ancestor of this baseline. Native
lessons require its installed `metatomic-lj-plugin.so`; a missing artifact produces
an explicit error. A released package is not assumed to expose the same API.

On this Mac the existing build configuration selected an older Cargo and an
incompatible C++ compiler. The following per-command overrides successfully
built the library using the already-installed Rust 1.88 and Apple Clang:

```sh
PATH="$HOME/.rustup/toolchains/1.88.0-aarch64-apple-darwin/bin:$PATH" \
CC=/usr/bin/clang CXX=/usr/bin/clang++ \
CARGO="$HOME/.rustup/toolchains/1.88.0-aarch64-apple-darwin/bin/cargo" \
RUSTC="$HOME/.rustup/toolchains/1.88.0-aarch64-apple-darwin/bin/rustc" \
uv sync --python 3.12
```

## Verify and export

```sh
uv run marimo check notebooks/*.py
uv run pytest -q
uv run python scripts/export_all.py
```

The tests execute **all 22** notebooks and independently check derivatives,
backend/native parity, labels, selection behavior, serialization and dynamics.
The export script runs every notebook again, saves HTML and per-notebook logs under
`exports/`, and records success/failure in `exports/manifest.json`. Generated files
are ignored by Git. See [Validation](VALIDATION.md) for the checked edition and the
precise limits of these checks.

The lab uses marimo's public ASGI API with one local Uvicorn worker on `127.0.0.1`.
Use `--port 2721` for another port and Ctrl+C to stop. The catalog lives in
`metatomic_marimo/catalog.json`. This launcher expects the source checkout beside
its notebooks; it is not a cloud service or an independently distributed wheel.

## Relationship to the published examples

The [official metatomic cookbook](https://docs.metatensor.org/metatomic/latest/examples/index.html)
was reviewed on 2026-10-06. This course follows its explain–implement–run progression
and extends it with three backends, independent derivative checks, visible source
and worked exercises. The published examples target the established Torch API;
those wrappers/loaders must not be silently substituted into the new core API.

| Published lesson | Extension here |
| --- | --- |
| [Export a model](https://docs.metatensor.org/metatomic/latest/examples/1-export-atomistic-model.html) | 02–04 calculations; 07 labeled prediction serialization; 10 explicitly provisional model packaging |
| [ASE dynamics](https://docs.metatensor.org/metatomic/latest/examples/2-running-ase-md.html) | 08 three-backend calculator, NVE conservation and linked trajectory |
| [Neighbor lists](https://docs.metatensor.org/metatomic/latest/examples/3-atomistic-model-with-nl.html) | 06 periodic topology, differentiable vectors and finite-difference checks |
| [Profiling](https://docs.metatensor.org/metatomic/latest/examples/4-profiling.html) | 09 batch parity, warm synchronized timing distribution and a named profiler region |
| [TorchSim introduction](https://docs.metatensor.org/metatomic/latest/examples/5-torchsim-getting-started.html) and [batching](https://docs.metatensor.org/metatomic/latest/examples/6-torchsim-batched.html) | 17 packed-system indexing and cell orientation; a production core adapter remains pending |
