# Checked edition — 2026-10-06

All **22 notebooks execute** in the pinned CPU environment. The independent
scientific suite passes **31 tests** (15.95 s), and all **22 notebooks export to HTML**.
Each chapter has a learning objective, prerequisites/scope, visible executed code,
a numerical or structural check, a worked exercise and primary references.
Integration chapters are finished design lessons with runnable fixtures; their
production engine migrations remain open in the [API status](ROADMAP.md).

## Environment and commands

macOS arm64; Python 3.12.10; metatomic-core 0.2.0.dev1213+git.b02b9ff;
marimo 0.25.1; NumPy 2.5.3; JAX/jaxlib 0.11.2; Torch 2.14.1;
ASE 3.29.0; chemiscope 1.1.0. The exact sibling revision and native build steps
are in the [guide](README.md); the Python dependency resolution is in `uv.lock`.

```sh
uv run marimo check notebooks/*.py
uv run ruff check --isolated --select E4,E7,E9,F metatomic_marimo notebooks scripts tests
uv run pytest -q
uv run python scripts/export_all.py
```

Marimo validation and isolated Ruff correctness checks pass. Ruff is a development
check available in this workspace; it is not required to run the notebooks.
`exports/manifest.json` records every HTML export and its log; all 22 entries pass.
The generated artifacts are ignored by Git. Static HTML retains saved output and
source but cannot rerun Python controls.

## Acceptance evidence by notebook

| Chapter | Calculation or property checked |
| --- | --- |
| 00 System | NumPy/JAX/Torch geometry, shape/dtype metadata and copied/shared storage behavior |
| 01 Metadata and pairs | Half-list displacement (1.2, 0, 0), labeled pair values and custom data |
| 02 NumPy | E=0.2 eV, forces ±2 eV/Å and stress xx=0.0192 eV/Å³; independent derivatives |
| 03 JAX | JIT System pytree; forces and all nine strain components checked, including periodic images |
| 04 Torch | Fresh autograd leaves; eager/aot_eager energy, force and stress parity |
| 05 Relaxation | Monotone energy, force convergence below 10⁻⁶ eV/Å and center-of-mass preservation |
| 06 Pair topology | Four directed periodic pairs, reconstructed vectors, finite-difference forces/cell stress |
| 07 Labeled outputs | Energy and position/strain gradient labels; prediction serialization round trip |
| 08 ASE dynamics | Three-backend energy/force/Voigt stress; 60-step NVE error below 10⁻⁴ eV and trajectory parity |
| 09 Profiling | 24-system batch agrees with scalar/analytic and Torch results; synchronized timing distribution and named profiler region |
| 10 API design | Five JSON object round trips, expected errors, manifest validation and three prototype backends |
| 11 LAMMPS | W=−Vσ and translation invariance for a force-balanced pair |
| 12 GROMACS | Force/stress dimensions and virtual-work invariance under unit conversion |
| 13 PLUMED | All Cartesian restraint-force components checked by central differences |
| 14 ASE state | Cached single-point result is invalidated after a position changes |
| 15 i-PI | Energy, force and extensive virial atomic-unit conversion round trips |
| 16 Chemiscope | Structure/atom targets, immutable geometry and numerical stress/force data |
| 17 TorchSim | Packed per-system energy sums; off-diagonal cell exposes required transpose |
| 18 Native LJ | Real plugin load, capabilities, requested pairs, energy gradients and force balance |
| 19 NumPy LJ | Analytic radial derivative versus finite differences; native labeled output parity and cutoff plots |
| 20 Torch LJ | Eager/compiled energy and gradient parity; NumPy and native references |
| 21 C++ contract | Three-backend values, gradients and labels; selections, periodic images, skin pairs and lifetime behavior |

The independent suite varies off-axis and periodic geometries, rather than relying
only on the displayed default dimer. It also checks empty requests, invalid loader
options, unsupported strain requests, output survival after native model close and
the known omitted-system selection failure. The short integration fixtures contain
inline assertions exercised by both notebook execution and HTML export.

## Boundaries of this validation

The native C++ LJ does not advertise strain gradients: **no native stress parity**
is claimed. Its checked system-energy path rejects selections omitting an entire
input system. The regression test records the failure; upstream code is unchanged.
The private tutorial bridge does not establish public plugin API completeness.

Chapter 06 uses shifted-force LJ; 18–21 use the reference's energy-shifted LJ.
The latter has a force discontinuity at cutoff. The pair fixture uses an 8 Å cell,
so the 3.2 Å skin pair has no alternate periodic image inside the 3 Å cutoff.

The examples are CPU teaching systems, not GPU performance benchmarks or production
engine compatibility tests. `aot_eager` validates graph capture/autograd. The external
LAMMPS/GROMACS/PLUMED/i-PI/TorchSim programs are not invoked. ASE emits non-failing
NumPy 2.5 shape deprecation warnings. No implementation commits were pushed.

## Live presentation checks

The lab separates tutorials, reference models, integration design notes and API
design; topic search filters the catalog. The guide and API status render as HTML.
The launcher selects Matplotlib's Agg backend before importing pyplot because ASGI
notebook kernels run in worker threads; a macOS GUI backend cannot render there.

Live browser checks visited all 22 notebook routes and waited for every displayed
computation cell to render (108 code blocks in total). Each route rendered without
a notebook exception. Visual inspection confirmed JAX equation typesetting and
code indentation, and the LJ energy/force plots; the lab's search returned the two virial lessons for “virial”.
The JAX separation control was changed from 1.20 to 1.22 Å: energy updated from
0.200000 to 0.242000 eV, and the independent strain-agreement assertion completed.
The original control was restored. Browser observations are recorded in
`exports/browser-review.json`; the test report is `exports/pytest.xml`.
