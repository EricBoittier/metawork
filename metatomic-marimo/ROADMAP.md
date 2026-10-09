# API status and implementation roadmap

Reviewed 2026-10-06 against core Python revision `b02b9ff3` and the accepted C++ LJ
source at `12b24b14`. This is the **local metawork tracker**. A finished teaching
notebook establishes the stated calculation or design fixture, not completion of
the associated production API or engine migration.

## Python API

| Requested task | Evidence in this course | Remaining implementation |
| --- | --- | --- |
| Error handling (Guillaume) | 10 catches Python/native errors and reproduces the incorrect native success status | Preserve original status; callback, thread and lifetime failure coverage |
| JSON classes and serialization (Guillaume) | 10 round-trips four top-level classes plus nested References | Cross-language malformed-input, schema evolution and validation coverage |
| Wrap System (Eric Boittier) | 00–01 run three array backends, ownership, pair/custom data | Maintain binding and tracing regression coverage |
| Execute models not defined in Python | 18/21 execute the real C++ plugin via an experimental private bridge | Supported public wrapper, ownership and cleanup on all failure paths |
| Generic custom-model base | 10 has a working abstract prototype; 21 maps it to BaseModel/ExternalModel | Public Python callback trampoline, owner retention and exception recovery |
| Python plugin and loader | 10 exercises a versioned local manifest; 21 separates load_plugin from load_model | Registration/dispatch and clean-process artifact loading through the real API |
| TorchModel | 04/20 verify eager and fullgraph aot_eager values/derivatives | Public base, nn.Module/state/device contract, existing-model compatibility |
| Custom Torch tutorial | 04 harmonic derivatives; 20 native-compatible LJ kernel | Migrate authoring/export examples to the public class when available |
| NumpyModel | 02 analytic harmonic class; 19/21 labeled LJ parity | Public subclass, callback registration and full capability validation |
| Custom NumPy tutorial | 02 derivation, 07 outputs, 19 shifted LJ and finite differences | Replace local wrappers with the public model interface |
| JaxModel (optional/later) | 03 differentiates today's System pytree; 21 has native-compatible LJ | Pytree model state, topology/tracing, serialization and dependency policy |
| Custom JAX tutorial | 03 JIT, forces and full strain; 06 periodic pair vectors | Public packaging/registration once supported |
| NumPy/Torch/JAX engine tutorial | 05 relaxation; 08 three-backend ASE dynamics; 09 batching | General capability negotiation, loader integration and production adapters |

## Integrations and reference models

| Workstream | Completed teaching evidence | Remaining production work |
| --- | --- | --- |
| LAMMPS pair_style/fix/compute/docs | 11 checks stress/virial sign and translation invariance; specifies each interface | Serial native LJ, owned/ghost mapping, MPI accumulation, restart and documentation |
| GROMACS MD module | 12 verifies angstrom/eV to nm/kJ/mol scaling and virtual work | Native lifecycle, engine integration and regression trajectory |
| PLUMED Action | 13 derives restraint chain rule and checks every Cartesian component | Model-output derivatives, action lifecycle and bias/engine parity |
| ASE | 08 runs a teaching calculator; 14 demonstrates cache invalidation | General core loader/capabilities, state/device/unit/error handling |
| i-PI | 15 converts energy/force/virial to atomic units and back | Protocol, sockets, reconnection, units and end-to-end trajectory |
| Chemiscope | 16 renders atom/structure targets, forces, cell and stress with immutable display data | General adapter for labeled core predictions and tested selection mappings |
| TorchSim | 17 checks packed reductions and transposed skew-cell conventions | Core model/device adapter, pressure convention and engine trajectory tests |
| JAX-MD | 03/06 supply differentiation/topology background; feature branch exists | Separate production adapter; differentiable state, displacement/neighbor contract and trajectory tests |
| C/C++ LJ model and plugin | 18/21 load and execute the accepted artifact; values, position gradients and labels verified | Preserve known omitted-system selection issue; native strain is unsupported |
| Python NumPy LJ | 19/21 analytic kernel and native-compatible local output contract pass | Public NumpyModel/plugin registration and engine artifact tests |
| Python Torch LJ | 20/21 eager/compiled kernel and native-compatible local outputs pass | Public TorchModel/plugin registration, device/state/compile matrix |

## Design anchors and acceptance criteria

Use the accepted C++ names, typed metadata and ownership boundaries: `BaseModel`,
`ExternalModel`, internal `execute_inner`, validated `execute_model`, and distinct
`load_plugin`/`load_model`. The native model uses string options, ordered TensorMaps,
energy shifting, non-strict half lists and selected-endpoint weighting.

Before completing a feature branch, exercise two systems, selected and empty atoms,
empty requests, periodic images, skin pairs, cutoff behavior and outputs retained
after model destruction. For public plugins, require a fresh-process load/evaluate
example through the supported API. A local class registry cannot establish that.

For stress, add an explicit strain capability, return `dE/dstrain` with the correct
labels, and let the engine normalize by volume. Validate homogeneous deformation
of both coordinates and cell. The current native LJ cannot serve as a strain oracle.
For engine rewrites, verify unit/sign conventions, topology rebuilds and production
ownership/parallelism beyond the deliberately small notebook fixtures.

[Validation](VALIDATION.md) records the checked environment and notebook acceptance
checks. The [study guide](README.md) distinguishes the public API from local helpers.

## Feature branches on the user fork — 2026-10-06

Created and verified 25 new branches on `EricBoittier/metatomic`, all starting
from `python-wrap-system` at `b02b9ff3df6458cca21fa8690d229bbda97e6f78`.
No implementation commits, PRs, issues or upstream writes were made. System wrapping
and the reported-complete C LJ plugin were excluded. Error handling and JSON
serialization branches cover the remaining validation/fixes described above.

These are branches in the metatomic fork, including the integration workstreams;
changes owned by external engine repositories will require their corresponding
checkouts/forks when implementation starts. Dependent work will need the relevant
model/plugin changes incorporated as they land.

| Task | Feature branch |
| --- | --- |
| Error handling: finish native status and callback/lifetime coverage | [codex/python-error-handling](https://github.com/EricBoittier/metatomic/tree/codex/python-error-handling) |
| JSON classes: finish cross-language serialization and validation | [codex/python-json-serialization](https://github.com/EricBoittier/metatomic/tree/codex/python-json-serialization) |
| Execute non-Python models from Python Systems | [codex/python-native-model-execution](https://github.com/EricBoittier/metatomic/tree/codex/python-native-model-execution) |
| Generic Python custom-model base class | [codex/python-model-base](https://github.com/EricBoittier/metatomic/tree/codex/python-model-base) |
| Python plugin registration and load_model | [codex/python-plugin-loader](https://github.com/EricBoittier/metatomic/tree/codex/python-plugin-loader) |
| TorchModel, existing model compatibility and torch.compile | [codex/python-torch-model](https://github.com/EricBoittier/metatomic/tree/codex/python-torch-model) |
| Custom Torch model tutorial | [codex/tutorial-custom-torch-model](https://github.com/EricBoittier/metatomic/tree/codex/tutorial-custom-torch-model) |
| NumpyModel base class | [codex/python-numpy-model](https://github.com/EricBoittier/metatomic/tree/codex/python-numpy-model) |
| Custom NumPy model tutorial | [codex/tutorial-custom-numpy-model](https://github.com/EricBoittier/metatomic/tree/codex/tutorial-custom-numpy-model) |
| Optional JaxModel base class | [codex/python-jax-model](https://github.com/EricBoittier/metatomic/tree/codex/python-jax-model) |
| Custom JAX model tutorial | [codex/tutorial-custom-jax-model](https://github.com/EricBoittier/metatomic/tree/codex/tutorial-custom-jax-model) |
| NumPy/Torch/JAX engine tutorial with the real model API | [codex/tutorial-python-engines](https://github.com/EricBoittier/metatomic/tree/codex/tutorial-python-engines) |
| LAMMPS pair_style metatomic migration | [codex/lammps-pair-new-api](https://github.com/EricBoittier/metatomic/tree/codex/lammps-pair-new-api) |
| LAMMPS fix metatomic migration | [codex/lammps-fix-new-api](https://github.com/EricBoittier/metatomic/tree/codex/lammps-fix-new-api) |
| LAMMPS compute metatomic migration | [codex/lammps-compute-new-api](https://github.com/EricBoittier/metatomic/tree/codex/lammps-compute-new-api) |
| LAMMPS migration documentation | [codex/lammps-new-api-docs](https://github.com/EricBoittier/metatomic/tree/codex/lammps-new-api-docs) |
| GROMACS custom MD module migration | [codex/gromacs-new-api](https://github.com/EricBoittier/metatomic/tree/codex/gromacs-new-api) |
| PLUMED metatomic Action migration | [codex/plumed-new-api](https://github.com/EricBoittier/metatomic/tree/codex/plumed-new-api) |
| ASE integration migration | [codex/ase-new-python-api](https://github.com/EricBoittier/metatomic/tree/codex/ase-new-python-api) |
| i-PI integration migration | [codex/ipi-new-python-api](https://github.com/EricBoittier/metatomic/tree/codex/ipi-new-python-api) |
| Chemiscope integration migration | [codex/chemiscope-new-python-api](https://github.com/EricBoittier/metatomic/tree/codex/chemiscope-new-python-api) |
| TorchSim integration migration | [codex/torchsim-new-python-api](https://github.com/EricBoittier/metatomic/tree/codex/torchsim-new-python-api) |
| Python NumPy LJ model for engine development | [codex/lj-python-numpy](https://github.com/EricBoittier/metatomic/tree/codex/lj-python-numpy) |
| Python Torch LJ model for engine development | [codex/lj-python-torch](https://github.com/EricBoittier/metatomic/tree/codex/lj-python-torch) |
| JAX-MD integration with the new Python API | [codex/jax-md-new-python-api](https://github.com/EricBoittier/metatomic/tree/codex/jax-md-new-python-api) |
