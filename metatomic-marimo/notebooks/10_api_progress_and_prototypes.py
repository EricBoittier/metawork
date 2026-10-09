import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 10 · Python API progress, plans, and prototypes

    **API design · Design study with runnable probes**

    **Learning objective.** Audit implemented Python primitives and assess model/loader prototypes against the accepted C++ contract.

    **Prerequisites.** 00–09; read 21 for native execution.

    A working notebook for every item in the implementation checklist. **Verified** means demonstrated against this installed checkout; **prototype** means code defined here, not a public metatomic API; **planned** means an acceptance target, not completed implementation.

    Source baseline: `b02b9ff3df6458cca21fa8690d229bbda97e6f78`, reviewed 2026-10-06. The environment probe below reports the installed version. This notebook tracks tutorial work in **metawork**. No upstream code, remote issues, or GitHub state is changed.

    Run top to bottom: inspect today's API → exercise errors and JSON → explore a local model contract and three implementations → round-trip a manifest → plan native execution and engine integration. The earlier chapters remain the detailed physics tutorials.

    **Expected result.** JSON objects round-trip; three harmonic prototypes agree; absent public model symbols remain explicitly marked.

    The native-execution experiment in [21](/notebooks/cpp-lj/) now provides direct C++ evidence. The local harmonic classes below are deliberately small contract sketches; the accepted C++ interface is the reference for public naming and ownership.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import json
    import abc
    import numpy as np
    import jax
    import jax.numpy as jnp
    import torch
    import metatomic
    from metatomic import (
        System,
        Quantity,
        PairListOptions,
        References,
        ModelMetadata,
        ModelCapabilities,
    )

    jax.config.update("jax_enable_x64", True)
    mo.show_code(position="above")
    return (
        ModelCapabilities,
        ModelMetadata,
        PairListOptions,
        Quantity,
        References,
        System,
        abc,
        jax,
        jnp,
        json,
        metatomic,
        mo,
        np,
        torch,
    )


@app.cell
def _(metatomic, mo):
    api_symbols = [
        "MetatomicError",
        "References",
        "Quantity",
        "PairListOptions",
        "ModelMetadata",
        "ModelCapabilities",
        "System",
        "Model",
        "TorchModel",
        "NumpyModel",
        "JaxModel",
        "load_model",
    ]
    api_inventory = [
        {"symbol": name, "available": hasattr(metatomic, name)} for name in api_symbols
    ]
    _display = mo.vstack(
        [
            mo.md(f"Installed core version: **{metatomic.__version__}**"),
            mo.ui.table(api_inventory, selection=None),
        ]
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Checklist and dependency order
    Owners below are copied from your checklist, not assigned by this notebook. Tutorial coverage does not imply upstream implementation completion.

    | Item | Owner | Evidence / state | Next acceptance target |
    |---|---|---|---|
    | Error handling | Guillaume | Python validation and native exception exercised below; status issue reproduced | Correct native status; callback exception identity, thread isolation and lifetime tests |
    | Classes for all JSON objects | Guillaume | Four documented top-level objects plus nested References round-trip below | Cross-language fixtures, malformed input and schema evolution policy |
    | Wrap System | Eric Boittier | Present; chapters 00/01/03/04/06 | Broader lifetime/device tests; JAX attached-data limitations remain |
    | Execute non-Python models | — | No public core Python loader/executor in this checkout | Native plugin → Python System → labeled energy/derivatives |
    | Generic custom-model base | — | Local abstract prototype below | Agree request/output/ownership contract and implement real base |
    | Python plugin + loader | — | Local manifest registry below, not a plugin | Registration, artifact format, clean-process loading and callbacks |
    | TorchModel + compatibility/compile | — | Local Torch subclass below; chapter 04 compiles kernel | Existing-model adapter and real class; capture/eager derivative parity |
    | Custom Torch tutorial | — | 04 + prototype here | Replace local class with public TorchModel; save/load/evaluate |
    | NumpyModel | — | Local analytic subclass below | Real base registration, dtype and array ownership contract |
    | Custom NumPy tutorial | — | 02/07 + prototype here | Standard outputs and executable model round-trip |
    | JaxModel (possibly later) | — | Local JAX subclass below; System pytree in 03 | Decide optional dependency, parameters/state and tracing contract |
    | Custom JAX tutorial | — | 03/06 + prototype here | Real subclass and serialization; explicit topology boundary |
    | NumPy/Torch/JAX engine tutorial | — | 05/08/09 + one engine step below | Generic capability negotiation, real loader, neighbor rebuilds |

    **Suggested order:** error/JSON/System foundations → native execution and generic model contract → Python registration/loader → NumPy and Torch classes/tutorials → engine integration → optional JAX packaging. Some tutorial prototypes can proceed in parallel with implementation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Error handling — exercise both Python and native errors
    Catch expected errors at an engine or UI boundary, retain their message and cause, and report enough context to locate the failing request. Do not silently return zero energy/forces after failure. This cell deliberately triggers invalid input and missing native data; catching them is part of the lesson.

    **Progress:** `_status.py` translates native errors and contains callback exception preservation machinery. The following probes do not validate callback lifetimes or thread isolation.
    """)
    return


@app.cell
def _(PairListOptions, Quantity, System, metatomic, mo, np):
    demo_system = System(
        "angstrom",
        np.array([1, 1], dtype=np.int32),
        np.array([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]]),
        5 * np.eye(3),
        np.ones(3, dtype=bool),
    )

    def capture_expected_error(label, operation):
        try:
            operation()
        except (ValueError, TypeError, metatomic.MetatomicError) as error:
            return {
                "case": label,
                "exception": type(error).__name__,
                "message": str(error),
                "status": str(getattr(error, "status", None)),
            }
        raise AssertionError(f"Expected {label} to fail")

    error_examples = [
        capture_expected_error(
            "invalid cutoff", lambda: PairListOptions(cutoff=-1.0, full_list=False)
        ),
        capture_expected_error(
            "invalid sample kind",
            lambda: Quantity(name="energy", unit="eV", sample_kind="invalid"),
        ),
        capture_expected_error(
            "missing native data", lambda: demo_system.custom_data("tutorial::missing")
        ),
    ]
    mo.show_code(mo.ui.table(error_examples, selection=None), position="above")
    return (demo_system,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Finding and proposed fix (not applied upstream)
    On this baseline, the missing-data exception is `MetatomicError` with the correct message but `status` reports `MTA_SUCCESS`. In `_status.py`, `_get_exception(status)` overwrites its input with the return value of `mta_last_error`. Retrieving an error successfully is distinct from the original call succeeding.

    **Proposal:** preserve the original operation status in a separate variable; track the error-retrieval status independently; define what to expose when a null-pointer failure has no explicit status. Before merging a fix, test a failing native status call, a null result, nested callback exceptions, repeated failures, separate threads, and cleanup after exception ownership transfers. This notebook records evidence, not a fix or a complete error-handling certification.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## JSON classes — serialize the actual public objects
    The local `docs/src/core/reference/json-formats.rst` documents PairListOptions, Quantity, ModelMetadata and ModelCapabilities. References is a nested object with its own class. `to_dict`/`from_dict` are the current serialization API; use Python's `json` module for strings. System arrays and executable model weights are not these JSON metadata objects.

    Keep units, sample kinds, gradients, and the pair-cutoff encoding intact. PairListOptions stores the cutoff using its floating-point bit pattern in hexadecimal; let the class handle it.
    """)
    return


@app.cell
def _(
    ModelCapabilities,
    ModelMetadata,
    PairListOptions,
    Quantity,
    References,
    json,
    mo,
):
    energy_request = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    json_objects = {
        "Quantity": energy_request,
        "PairListOptions": PairListOptions(
            cutoff=3.5, full_list=False, requestors=["tutorial"]
        ),
        "References": References(model=["tutorial:original-harmonic-example"]),
        "ModelMetadata": ModelMetadata(
            name="Tutorial harmonic model",
            authors=["Tutorial author"],
            references=References(model=["tutorial:original-harmonic-example"]),
            extra={"scope": "teaching"},
        ),
        "ModelCapabilities": ModelCapabilities(
            outputs=[energy_request],
            atomic_types=[1],
            interaction_range=2.5,
            length_unit="angstrom",
            supported_devices=["cpu"],
            dtype="float64",
        ),
    }
    json_payloads = {
        name: json.dumps(obj.to_dict(), allow_nan=False, indent=2)
        for name, obj in json_objects.items()
    }
    json_roundtrips = {
        name: type(obj).from_dict(json.loads(json_payloads[name]))
        for name, obj in json_objects.items()
    }
    assert all(json_roundtrips[name] == obj for name, obj in json_objects.items())
    _display = mo.accordion(
        {
            name: mo.md("```json\n" + payload + "\n```")
            for name, payload in json_payloads.items()
        }
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Plan:** add Python↔Rust/C++ reference fixtures for every documented object, nested output quantities and references; test invalid type tags, missing fields, NaN/infinity and wrong field types. Decide whether unknown fields are rejected or retained before promising forward compatibility. Round-trip equality alone does not prove cross-language compatibility. Add any future JSON object to this inventory and its fixtures.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## System — the shared geometry boundary
    **Progress:** the System above holds NumPy arrays. [00 System](/notebooks/system/) constructs all three backends; [01 metadata](/notebooks/metadata/) attaches pair/custom data; [03 JAX](/notebooks/jax/) differentiates System leaves; [04 Torch](/notebooks/torch/) explains DLPack/autograd boundaries.

    **Next checks:** dtype/device rules, zero atoms, nonperiodic cell conventions, released handles, views, reference ownership, backend conversion and lifetime after temporary arrays disappear. JAX transformations currently reject attached pair/custom data; [06](/notebooks/neighbors/) passes topology separately and reconstructs vectors from traced geometry. Do not infer that zero-copy transport preserves a Torch autograd graph.
    """)
    return


@app.cell
def _(demo_system, mo):
    system_snapshot = {
        "atoms": len(demo_system),
        "backend": demo_system.arrays_backend,
        "length_unit": demo_system.length_unit,
        "positions_shape": list(demo_system.positions.shape),
        "cell_shape": list(demo_system.cell.shape),
    }
    mo.show_code(system_snapshot, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Generic model base — an executable contract sketch
    **Prototype, not `metatomic.Model`.** A deliberately small abstract interface makes engine/model responsibilities discussable. This example accepts exactly two H atoms in an orthogonal periodic box, angstrom geometry, CPU float64. It returns a local dictionary of energy/forces/stress; the production interface should use agreed Quantity requests and labeled outputs (see [07](/notebooks/outputs/)).

    Decisions still needed: batch versus single-system calls, requested outputs and explicit gradients, selected atoms, device negotiation, pair ownership, optional inputs, validation cost, and exception propagation. Our toy prototype rejects unsupported requests; it does not silently pretend to support the full future contract.
    """)
    return


@app.cell
def _(abc, mo, np):
    class PrototypeModel(abc.ABC):
        """Notebook-local interface; deliberately not the future public model API."""

        def __init__(self, k=10.0, r0=1.0):
            if not np.isfinite(k) or k <= 0 or not np.isfinite(r0) or r0 <= 0:
                raise ValueError("k and r0 must be positive finite numbers")
            self.k, self.r0 = float(k), float(r0)

        def pair(self, system):
            r, c = np.asarray(system.positions), np.asarray(system.cell)
            if system.length_unit != "angstrom" or r.dtype != np.float64:
                raise ValueError("prototype requires angstrom and float64")
            if (
                len(system) != 2
                or not np.all(system.types == 1)
                or not np.all(system.pbc)
            ):
                raise ValueError("prototype supports two periodic hydrogen atoms")
            if not np.allclose(c, np.diag(np.diag(c))) or np.any(np.diag(c) <= 0):
                raise ValueError("prototype requires a positive orthogonal cell")
            if not np.all(np.isfinite(r)) or not np.all(np.isfinite(c)):
                raise ValueError("geometry must be finite")
            d = r[1] - r[0]
            d = d - np.round(d @ np.linalg.inv(c)) @ c
            if np.linalg.norm(d) == 0:
                raise ValueError("coincident atoms")
            return d, abs(np.linalg.det(c))

        @abc.abstractmethod
        def radial(self, d):
            """Return energy and derivative with respect to minimum-image vector."""
            raise NotImplementedError

        def evaluate(self, system, outputs=("energy", "forces", "stress")):
            if not set(outputs) <= {"energy", "forces", "stress"}:
                raise ValueError("unsupported requested output")
            d, volume = self.pair(system)
            energy, derivative = self.radial(d)
            results = {
                "energy": float(energy),
                "forces": np.stack([derivative, -derivative]),
                "stress": np.outer(d, derivative) / volume,
            }
            return {name: results[name] for name in outputs}

    mo.show_code(position="above")
    return (PrototypeModel,)


@app.cell
def _(mo):
    mo.md(r"""
    ## NumpyModel and custom NumPy tutorial
    **Prototype:** subclass the local interface with an analytic derivative. Compare it to finite differences before teaching an engine to trust it. [02](/notebooks/numpy/) derives this potential; [07](/notebooks/outputs/) shows how energy derivatives become TensorMap gradients.

    **Plan for the real class:** pin array ownership/dtype semantics, validate declared capabilities, implement standard outputs, register through the Python plugin, then round-trip an executable artifact in a fresh process. Completion requires the real `NumpyModel`, not renaming this class.
    """)
    return


@app.cell
def _(PrototypeModel, demo_system, mo, np):
    class PrototypeNumpyModel(PrototypeModel):
        def radial(self, d):
            radius = np.linalg.norm(d)
            energy = 0.5 * self.k * (radius - self.r0) ** 2
            derivative = self.k * (radius - self.r0) * d / radius
            return energy, derivative

    numpy_prototype = PrototypeNumpyModel()
    prototype_prediction = numpy_prototype.evaluate(demo_system)
    mo.show_code(prototype_prediction, position="above")
    return PrototypeNumpyModel, numpy_prototype, prototype_prediction


@app.cell
def _(mo):
    mo.md(r"""
    ## TorchModel, compile compatibility, and custom Torch tutorial
    **Prototype:** delegate to an `nn.Module` tensor kernel and start a fresh autograd graph. This wrapper returns CPU NumPy results for the local engine contract; that conversion is not a production GPU design. [04](/notebooks/torch/) tests `torch.compile(fullgraph=True, backend="aot_eager")` against eager energy and derivatives. Compiling a tensor kernel does not prove that System transport or a plugin callback is compilable.

    **Plan:** establish the real TorchModel's relationship to `nn.Module`; keep tensor-only graph regions explicit; test parameter registration, state_dict, existing Torch models, device/dtype moves and compiled/eager gradients. Compatibility with existing `metatomic.torch` models needs an explicit adapter and independent tests.
    """)
    return


@app.cell
def _(PrototypeModel, demo_system, mo, torch):
    class PrototypeTorchKernel(torch.nn.Module):
        def __init__(self, k, r0):
            super().__init__()
            self.register_buffer("k", torch.tensor(k, dtype=torch.float64))
            self.register_buffer("r0", torch.tensor(r0, dtype=torch.float64))

        def forward(self, vector):
            return 0.5 * self.k * (torch.linalg.vector_norm(vector) - self.r0) ** 2

    class PrototypeTorchModel(PrototypeModel):
        def __init__(self, k=10.0, r0=1.0):
            super().__init__(k, r0)
            self.kernel = PrototypeTorchKernel(k, r0)

        def radial(self, d):
            vector = torch.tensor(d.copy(), dtype=torch.float64, requires_grad=True)
            energy = self.kernel(vector)
            derivative = torch.autograd.grad(energy, vector)[0]
            return energy.detach().item(), derivative.detach().numpy().copy()

    torch_prototype = PrototypeTorchModel()
    mo.show_code(torch_prototype.evaluate(demo_system), position="above")
    return PrototypeTorchModel, torch_prototype


@app.cell
def _(mo):
    mo.md(r"""
    ## JaxModel (optional later) and custom JAX tutorial
    **Prototype:** JIT a pure vector function and differentiate it. Parameters here are captured at construction; mutating `k` afterwards would not update the compiled closure. A real trainable model should pass parameter/state pytrees explicitly.

    **Plan:** decide immutable parameters versus mutable runtime state; define static metadata, batching, topology handling and shape specialization; avoid Python/C callbacks inside tracing. [03](/notebooks/jax/) differentiates System geometry directly and [06](/notebooks/neighbors/) handles external pair topology. Test parameter serialization plus fresh-process reconstruction before promising JAX executable portability.
    """)
    return


@app.cell
def _(
    PrototypeModel,
    demo_system,
    jax,
    jnp,
    mo,
    np,
    numpy_prototype,
    prototype_prediction,
    torch_prototype,
):
    class PrototypeJaxModel(PrototypeModel):
        def __init__(self, k=10.0, r0=1.0):
            super().__init__(k, r0)
            self.value_gradient = jax.jit(
                jax.value_and_grad(
                    lambda vector: 0.5 * k * (jnp.linalg.norm(vector) - r0) ** 2
                )
            )

        def radial(self, d):
            energy, derivative = self.value_gradient(jnp.asarray(d))
            return float(energy), np.array(derivative)

    jax_prototype = PrototypeJaxModel()
    prototype_models = {
        "numpy": numpy_prototype,
        "torch": torch_prototype,
        "jax": jax_prototype,
    }
    prototype_parity = []
    for _backend, _model in prototype_models.items():
        _prediction = _model.evaluate(demo_system)
        for _key in prototype_prediction:
            np.testing.assert_allclose(
                _prediction[_key], prototype_prediction[_key], atol=1e-12
            )
        prototype_parity.append(
            {
                "backend": _backend,
                "energy_eV": _prediction["energy"],
                "max_force": float(
                    np.max(np.linalg.norm(_prediction["forces"], axis=1))
                ),
                "stress_xx": float(_prediction["stress"][0, 0]),
            }
        )
    mo.show_code(mo.ui.table(prototype_parity, selection=None), position="above")
    return PrototypeJaxModel, prototype_models


@app.cell
def _(mo):
    mo.md(r"""
    ## Python plugin + loader — local manifest experiment
    **Prototype only:** a JSON manifest chooses one of three explicitly registered local classes and their harmonic parameters. No arbitrary import path, pickle, native plugin or weights are loaded. This makes versioning and reconstruction concrete without inventing a `metatomic.load_model` API.

    **Production plan:** agree artifact format, backend identifier/version, parameters/weights and metadata; implement Python registration and callback ownership; propagate errors through the native boundary; test clean-process load, missing dependencies, unsupported format versions and deterministic predictions. This tiny manifest cannot save arbitrary models.
    """)
    return


@app.cell
def _(
    PrototypeJaxModel,
    PrototypeNumpyModel,
    PrototypeTorchModel,
    demo_system,
    json,
    mo,
    np,
    prototype_prediction,
):
    prototype_registry = {
        "numpy": PrototypeNumpyModel,
        "torch": PrototypeTorchModel,
        "jax": PrototypeJaxModel,
    }

    def load_prototype(text):
        manifest = json.loads(text)
        if not isinstance(manifest, dict) or set(manifest) != {
            "format",
            "version",
            "backend",
            "parameters",
        }:
            raise ValueError("invalid prototype manifest fields")
        if (
            manifest["format"] != "tutorial-harmonic"
            or type(manifest["version"]) is not int
            or manifest["version"] != 1
        ):
            raise ValueError("unsupported prototype format/version")
        backend = manifest["backend"]
        if not isinstance(backend, str) or backend not in prototype_registry:
            raise ValueError("unregistered prototype backend")
        parameters = manifest["parameters"]
        if not isinstance(parameters, dict) or set(parameters) != {"k", "r0"}:
            raise ValueError("expected k and r0")
        if any(
            type(value) not in (int, float) or not np.isfinite(value) or value <= 0
            for value in parameters.values()
        ):
            raise ValueError("parameters must be finite positive numbers")
        return prototype_registry[backend](**parameters)

    manifest_text = json.dumps(
        {
            "format": "tutorial-harmonic",
            "version": 1,
            "backend": "numpy",
            "parameters": {"k": 10.0, "r0": 1.0},
        },
        allow_nan=False,
        indent=2,
    )
    reloaded_prototype = load_prototype(manifest_text)
    np.testing.assert_allclose(
        reloaded_prototype.evaluate(demo_system)["forces"],
        prototype_prediction["forces"],
    )
    mo.show_code(mo.md("```json\n" + manifest_text + "\n```"), position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Execute a model not defined in Python
    **Current boundary:** core Systems exist, but the public core Python namespace lacks a model loader/executor. The manifest experiment above reconstructs Python classes and therefore does **not** complete this task.

    **Working evidence:** [18](/notebooks/lj-c/) loads the accepted C++ LJ artifact and [21](/notebooks/cpp-lj/) checks native/backend values, position gradients and labels through an experimental private bridge. The installed plugin supplies the non-Python reference; its strain capability is absent.

    **Public-API acceptance test:** use the supported loader and executor to inspect capabilities, construct a System, fulfill pair requests and recover labeled energy/position gradients. Compare forces with finite differences and NumPy. Repeat in a fresh process and test missing symbols, incompatible ABI, model errors and library/array lifetimes. Add a separate strain-capable reference before testing stress output.

    **Design questions:** where is plugin discovery configured, who owns model handles, which ABI versions are accepted, and what happens when a model outlives its shared library? Keep the library alive until its model and all callback-owned outputs are released. Treat the bridge as a contract experiment; migrate application code to the supported wrapper when it exists.

    The local C API and native model interfaces can inform this design, but their existence does not imply a shipped Python execution wrapper.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Engine tutorial — consume the same contract from all three backends
    **Runnable precursor:** one fixed-cell relaxation step uses the common local interface. [05](/notebooks/engine/) implements convergence; [08](/notebooks/dynamics/) runs ASE NVE with three backends and stress; [09](/notebooks/profiling/) explores batching. These examples already separate geometry evolution from energy evaluation.

    **Production plan:** load model → negotiate capabilities/units/device/dtype → build requested pair lists → request outputs → interpret labeled gradients → update geometry → rebuild pairs as needed. Cache topology only with a documented skin/rebuild policy. Handle errors without integrating invalid forces. Variable-cell integration must respect the stress sign and volume convention.
    """)
    return


@app.cell
def _(System, demo_system, mo, prototype_models):
    engine_step_results = []
    for _backend, _model in prototype_models.items():
        _before = _model.evaluate(demo_system)
        _updated = System(
            "angstrom",
            demo_system.types.copy(),
            demo_system.positions + 0.01 * _before["forces"],
            demo_system.cell.copy(),
            demo_system.pbc.copy(),
        )
        _after = _model.evaluate(_updated)
        assert _after["energy"] < _before["energy"]
        engine_step_results.append(
            {
                "backend": _backend,
                "before_eV": _before["energy"],
                "after_eV": _after["energy"],
            }
        )
    mo.show_code(mo.ui.table(engine_step_results, selection=None), position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Definition of done and next experiments
    1. **Foundations:** resolve native error status; round-trip every current JSON schema across languages; document System ownership/backend limits.
    2. **Execution:** a real non-Python plugin evaluates Python Systems using a public loader with tested lifecycle/error propagation.
    3. **Custom models:** actual generic/NumPy/Torch classes pass the same contract suite; optional JAX has explicit state/tracing semantics.
    4. **Packaging:** save and load a real model in a clean process, preserving metadata, capabilities and predictions.
    5. **Engines/tutorials:** requests, neighbor rebuilding, labels, units and gradients work end to end; eager/compiled and backend agreement are checked on more than one geometry.

    **Try it:** change manifest backend and stiffness, then compare all predictions; inject an unsupported manifest version and inspect the error; add a translated and periodic-image test; replace the local dictionary result with chapter 07's TensorMap representation. Avoid marking upstream work complete because its notebook prototype passes.

    ### Evidence and further reading
    Local source: `python/metatomic_core/src/metatomic/{__init__,_status,_system,_quantity,_metadata,_capabilities}.py`; schema reference: `docs/src/core/reference/json-formats.rst`. The notebook prints installed API availability, while the analysis describes the pinned source baseline.

    [Published cookbook](https://docs.metatensor.org/metatomic/latest/examples/index.html) · [Local API roadmap](/roadmap) · [Outputs and contract](/notebooks/outputs/). Published Torch wrappers and this emerging core API are distinct; translate concepts deliberately.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    Does the working manifest loader establish that the public Python plugin API is complete?

    **Worked answer.** A manifest that reconstructs a Python class is not a generic plugin loader. Completion requires the real callback/ownership boundary and a fresh-process load/evaluate test. Use BaseModel/ExternalModel/execute_model as C++ naming anchors.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce and continue
    Run `uv run marimo edit notebooks/10_api_progress_and_prototypes.py` from this project. Use the default controls to reproduce the reference result, then vary one input at a time. Replace `.value` controls with numeric values when copying to a script. Computation cells show their executed source; plotting helpers belong to this cookbook.

    This edition targets core System revision `b02b9ff3`, Python 3.12 and CPU float64. Source and dependencies are pinned in the guide; HTML exports preserve results but cannot rerun Python controls.

    [Accepted C++ model interface](https://github.com/metatensor/metatomic/blob/12b24b14/metatomic-core/include/metatomic/model.hpp) · [Guide](/guide) · [API status](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
