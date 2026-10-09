import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 21 · Use the upstream C++ LJ model as the reference

    **Reference models · Executable tutorial**

    **Learning objective.** Execute the accepted C++ LJ plugin and compare values, gradients and labels with three Python kernels.

    **Prerequisites.** 01, 06–07 and one backend chapter.

    This chapter executes the installed C++ LJ plugin through the C API and compares matching NumPy/Torch/JAX implementations. The Python reference models and native bridge live in `metatomic_marimo.lj_reference`. They establish design requirements for the pending public model classes and loader.

    Reviewed upstream `metatomic-core` at `12b24b14` (2026-10-06). That commit is already an ancestor of our `python-wrap-system` baseline `b02b9ff`; updating to the upstream branch alone would discard the additional Python System work.

    Source anchors: [LJ implementation](https://github.com/metatensor/metatomic/blob/12b24b14/metatomic-core/lj-plugin/lennard_jones.cpp), [model interface](https://github.com/metatensor/metatomic/blob/12b24b14/metatomic-core/include/metatomic/model.hpp), [plugin interface](https://github.com/metatensor/metatomic/blob/12b24b14/metatomic-core/include/metatomic/plugin.hpp), [LJ tests](https://github.com/metatensor/metatomic/blob/12b24b14/metatomic-core/tests/cxx/lj-plugin.cpp).

    **Expected result.** For σ = ε = 1 and rc = 3, E(2 Å) ≈ −0.056044 eV; backend parity errors are near float64 precision.
    """)
    return


@app.cell
def _():
    import inspect
    import json
    import marimo as mo
    import numpy as np
    import jax
    from metatomic import Quantity
    from metatomic_marimo.lj_reference import (
        DEFAULT_OPTIONS,
        NativeLJ,
        ReferenceLJ,
        NumpyLJ,
        TorchLJ,
        JaxLJ,
        make_pair_system,
        labels,
    )

    jax.config.update("jax_enable_x64", True)
    mo.show_code(position="above")
    return (
        DEFAULT_OPTIONS,
        JaxLJ,
        NativeLJ,
        NumpyLJ,
        Quantity,
        ReferenceLJ,
        TorchLJ,
        inspect,
        json,
        labels,
        make_pair_system,
        mo,
        np,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Adopt the established contract and style
    | Upstream convention | Consequence for Python work |
    |---|---|
    | `BaseModel` abstract interface; `ExternalModel` owns a native model | Keep authoring and loaded-model ownership separate; do not assume the final public name is `Model`. |
    | `capabilities`, `metadata`, `requested_pair_lists`, `requested_inputs` | Mirror the existing method names and typed JSON objects. |
    | `execute_inner(systems, selected_atoms, requested_outputs)` | Batch input; optional selection; one TensorMap per request, in request order. |
    | Validated `execute_model(..., check_consistency)` | Engines call the executor; direct callbacks are an internal implementation boundary. |
    | `load_plugin(path)` then `load_model(load_from, options, plugin_name)` | Keep plugin registration distinct from model construction and lifetime. |
    | String-to-string loader options | Do not replace the plugin protocol with numeric JSON parameters without a design decision. |

    The useful style guide here is the accepted code: small focused methods, explicit ownership, narrow helpers, typed metadata, consistent names, and tests of observable behavior. These are source-derived conventions, not guesses about an individual reviewer's preferences.

    ## Match the potential, not just the class names
    The reference is **energy-shifted**: `U(r) = 4 epsilon [(sigma/r)^12 - (sigma/r)^6] - U_LJ(cutoff)` below cutoff; zero at and above cutoff. It is not force-shifted. The force is discontinuous at cutoff, so finite-difference checks avoid that point.

    It requests a **half list**, `full_list=False`, `strict=False`: the engine may supply skin pairs, and the model excludes distances at/above cutoff. System energy supports position gradients; atom energy has no gradients. **Strain/stress is not advertised.** Notebook 06's force-shifted full-list example remains a separate teaching choice, not a parity reference.
    """)
    return


@app.cell
def _(Quantity, make_pair_system, mo):
    reference_systems = [
        make_pair_system(2.0),
        make_pair_system(1.2, transverse=0.2, image=True),
        make_pair_system(3.2),
    ]
    reference_requests = [
        Quantity(
            name="energy", unit="eV", sample_kind="system", gradients=["positions"]
        ),
        Quantity(name="energy", unit="eV", sample_kind="atom"),
    ]
    mo.show_code(reference_requests, position="above")
    return reference_requests, reference_systems


@app.cell
def _(mo):
    mo.md(r"""
    ## Execute the actual C++ plugin from Python
    This experiment uses the existing **private Python ctypes bindings** to the public C functions. It requires the plugin installed by this pinned local build. It is deliberately scoped to the notebook, rather than exported as a supported loader.

    Observe the `with` block: model ownership is explicit and unload runs once. Returned TensorMaps own their output allocations independently. Registered plugin libraries remain managed by the core. A `metatensor::SimpleDataArray` wrapper is registered so Python can read the native arrays; the wrapper retains the output owner.

    The concrete next step for the execution feature branch is to replace this experimental bridge with a reviewed generic `ExternalModel`/executor wrapper and dedicated ownership/error tests.
    """)
    return


@app.cell
def _(
    DEFAULT_OPTIONS,
    NativeLJ,
    json,
    mo,
    reference_requests,
    reference_systems,
):
    with NativeLJ(DEFAULT_OPTIONS) as _native:
        cpp_capabilities = _native.capabilities()
        cpp_metadata = _native.metadata()
        cpp_pairs = _native.requested_pair_lists()
        cpp_outputs = _native.execute(reference_systems, None, reference_requests)
    cpp_energies = cpp_outputs[0].block().values.copy()
    _display = mo.vstack(
        [
            mo.md("**Real C++ plugin results:**"),
            mo.md(str(cpp_metadata)),
            mo.md("Requested pairs: " + repr(cpp_pairs)),
            cpp_energies,
            mo.md(
                "```json"
                + chr(10)
                + json.dumps(cpp_capabilities.to_dict(), indent=2)
                + chr(10)
                + "```"
            ),
        ]
    )
    mo.show_code(_display, position="above")
    return cpp_energies, cpp_outputs


@app.cell
def _(NativeLJ, inspect, mo):
    _display = mo.accordion(
        {
            "Native execution bridge — actual source": mo.md(
                "```python" + chr(10) + inspect.getsource(NativeLJ) + chr(10) + "```"
            )
        }
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## NumPy, Torch, JAX — change the kernel, retain output semantics
    The reference classes share metadata, request validation, pair accounting and labeled output construction. NumPy uses analytic derivatives; Torch uses autograd; JAX uses `jit(vmap(value_and_grad))`. The engine boundary is CPU NumPy geometry and outputs for all three, which keeps the comparison explicit. This is not a GPU adapter or a complete backend model base.

    Attached topology stays outside JAX transformations. The kernel differentiates pair vectors; the adapter scatters their derivatives back to both atoms. The final public classes must resolve tensor ownership, device/state and plugin callbacks separately.
    """)
    return


@app.cell
def _(
    JaxLJ,
    NumpyLJ,
    TorchLJ,
    cpp_energies,
    cpp_outputs,
    mo,
    np,
    reference_requests,
    reference_systems,
):
    backend_references = {"numpy": NumpyLJ(), "torch": TorchLJ(), "jax": JaxLJ()}
    backend_comparison = []
    for _name, _model in backend_references.items():
        _outputs = _model.execute(reference_systems, None, reference_requests)
        for _actual, _expected in zip(_outputs, cpp_outputs, strict=True):
            assert _actual.keys == _expected.keys
            assert _actual.block().samples == _expected.block().samples
            np.testing.assert_allclose(
                _actual.block().values, _expected.block().values, atol=1e-12
            )
        _gradient = _outputs[0].block().gradient("positions")
        _cpp_gradient = cpp_outputs[0].block().gradient("positions")
        assert _gradient.samples == _cpp_gradient.samples
        np.testing.assert_allclose(_gradient.values, _cpp_gradient.values, atol=1e-11)
        backend_comparison.append(
            {
                "backend": _name,
                "max_energy_error": float(
                    np.max(np.abs(_outputs[0].block().values - cpp_energies))
                ),
                "max_gradient_error": float(
                    np.max(np.abs(_gradient.values - _cpp_gradient.values))
                ),
            }
        )
    mo.show_code(mo.ui.table(backend_comparison, selection=None), position="above")
    return (backend_references,)


@app.cell
def _(JaxLJ, NumpyLJ, ReferenceLJ, TorchLJ, inspect, make_pair_system, mo):
    _display = mo.accordion(
        {
            name: mo.md(
                "```python" + chr(10) + inspect.getsource(cls) + chr(10) + "```"
            )
            for name, cls in {
                "Shared reference contract": ReferenceLJ,
                "NumPy kernel": NumpyLJ,
                "Torch kernel": TorchLJ,
                "JAX kernel": JaxLJ,
                "Engine pair fixture": make_pair_system,
            }.items()
        }
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Atom selection is an energy partition, not deletion of neighbors
    Each selected endpoint contributes half the pair energy. A system energy with one selected endpoint is therefore half that pair energy, but its gradient still contains both atoms. Keep the selected atom's environment. For atom energy, samples preserve the selected atom labels and order.

    The batch also tests a periodic-image vector and a skin pair beyond cutoff. The C++ result is the reference; the Python implementations must agree in both values and labels.
    """)
    return


@app.cell
def _(
    DEFAULT_OPTIONS,
    NativeLJ,
    backend_references,
    labels,
    mo,
    np,
    reference_requests,
    reference_systems,
):
    selected_atoms = labels(["system", "atom"], [[1, 1], [0, 0], [2, 0]])
    with NativeLJ(DEFAULT_OPTIONS) as _native:
        selected_cpp = _native.execute(
            reference_systems, selected_atoms, reference_requests
        )
    for _model in backend_references.values():
        _outputs = _model.execute(reference_systems, selected_atoms, reference_requests)
        for _actual, _expected in zip(_outputs, selected_cpp, strict=True):
            assert _actual.block().samples == _expected.block().samples
            np.testing.assert_allclose(
                _actual.block().values, _expected.block().values, atol=1e-12
            )
        np.testing.assert_allclose(
            _outputs[0].block().gradient("positions").values,
            selected_cpp[0].block().gradient("positions").values,
            atol=1e-11,
        )
    mo.show_code(selected_cpp[0].block().values, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Fill the feature branches in dependency order
    | Branch/work item | Design now grounded in the C++ code | Remaining public-API work |
    |---|---|---|
    | Native execution | Owned external handle, metadata callbacks, C executor, ordered TensorMaps | Generic wrapper; output cleanup on all failure paths; optional callbacks; lifetime and conversion tests |
    | Generic model base | Five abstract methods, internal `execute_inner`, engine-facing validated executor | Python callback trampoline and owner retention; recover original exceptions |
    | Python plugin/loader | `load_plugin` separate from `load_model`; string-valued options; unsupported-model signal | Registration, dispatch and clean-process import/artifact contract |
    | NumpyModel | NumPy reference now agrees with real C++ outputs | Public subclass and callback registration; full capability/input validation |
    | TorchModel | Autograd kernel agrees; keep transport outside compiled regions | `nn.Module` relationship, existing-model adapter, state/device and compile matrix |
    | JaxModel | Pure pair kernel agrees with explicit parameters | Pytree state, topology/tracing limits, serialization and optional dependency policy |
    | NumPy/Torch LJ models | Energy-shift, half-list and selection conventions are concrete | Real model wrappers and artifact/engine tests |

    **Do not mark these branches complete yet.** The working code is in the local tutorial project. The experiments establish numerical and ownership requirements; they do not implement the full public callback/plugin API.

    ### Stress extension
    If adding stress to LJ, explicitly add the strain capability, return `dE/dstrain` with correct labels and convert to stress by volume in engines. Validate the same deformation of both coordinates and cell against finite differences. The C++ model currently cannot serve as a strain-output oracle.

    ### Review checklist
    Match method names and options; preserve error messages and unsupported-model behavior; return independent owned outputs; test two systems, selected/empty atoms, empty requests, periodic images, skin pairs and cutoff; keep backend-specific logic in kernels. Do not replace the accepted potential with a smoother one merely to make derivatives easier.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Observed selection edge case in the accepted reference
    Checked system-energy execution currently raises `invalid samples` when a selection omits an entire input system, including an empty selection. The LJ callback creates samples for every system; the core consistency validator expects samples corresponding to the selection. The passing selection example above includes an atom from every input system.

    A regression probe records the failure rather than claiming parity for an invalid result. Resolve the intended semantics before porting this path: either return only selected-system samples and remap gradient sample indices, or agree a different core contract. The upstream C++ tests select atoms from every system, so they do not exercise this gap. No upstream fix is applied here.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    If you select one endpoint of a pair, can you omit the other atom from the force calculation?

    **Worked answer.** Selecting one endpoint weights its pair energy by 1/2 but does not remove the other endpoint from the derivative. The current native checked system-energy path rejects selections omitting an entire input system; preserve this as a known limitation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce and continue
    Run `uv run marimo edit notebooks/21_cpp_lj_contract.py` from this project. Use the default controls to reproduce the reference result, then vary one input at a time. Replace `.value` controls with numeric values when copying to a script. Computation cells show their executed source; plotting helpers belong to this cookbook.

    This edition targets core System revision `b02b9ff3`, Python 3.12 and CPU float64. Source and dependencies are pinned in the guide; HTML exports preserve results but cannot rerun Python controls.

    [Accepted LJ tests](https://github.com/metatensor/metatomic/blob/12b24b14/metatomic-core/tests/cxx/lj-plugin.cpp) · [Guide](/guide) · [API status](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
