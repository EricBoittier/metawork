"""LJ contract experiments modeled on upstream 12b24b14, not public API wrappers.

NativeLJ uses private Python bindings to the real public C API for this pinned
checkout only. NumPy/Torch/JAX classes implement the same small reference contract.
"""

import ctypes
import json
import threading
from pathlib import Path

import metatomic
import numpy as np
from metatomic import (
    ModelCapabilities,
    ModelMetadata,
    PairListOptions,
    Quantity,
    System,
)
from metatensor import Labels, TensorBlock, TensorMap


DEFAULT_OPTIONS = {
    "sigma": "1",
    "epsilon": "1",
    "cutoff": "3",
    "atomic_type": "1",
    "length_unit": "angstrom",
    "energy_unit": "eV",
}
_PLUGIN_LOCK = threading.Lock()
_PLUGIN_LOADED = False


def labels(names, rows):
    return Labels(names, np.asarray(rows, dtype=np.int32).reshape(-1, len(names)))


def pair_options(cutoff=3.0):
    return PairListOptions(cutoff=cutoff, full_list=False, strict=False)


def make_pair_system(distance=1.2, transverse=0.0, image=False):
    """Two-atom test fixture with one engine-owned half-list pair.

    The 8 angstrom box isolates the pairs used here (up to 3.2 angstrom)
    from other images inside the 3 angstrom cutoff. This is not a pair search.
    """
    positions = np.array([[0.2, 1.0, 1.0], [0.2 + distance, 1.0 + transverse, 1.0]])
    shift = np.zeros(3, dtype=np.int32)
    if image:
        positions[1, 0] -= 8.0
        shift[0] = 1
    cell = 8.0 * np.eye(3)
    system = System(
        "angstrom",
        np.array([1, 1], dtype=np.int32),
        positions,
        cell,
        np.ones(3, dtype=bool),
    )
    vector = positions[1] - positions[0] + shift @ cell
    block = TensorBlock(
        vector.reshape(1, 3, 1),
        labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            [[0, 1, *shift]],
        ),
        [labels(["xyz"], [[0], [1], [2]])],
        labels(["distance"], [[0]]),
    )
    system.add_pairs(pair_options(), block)
    return system


class NativeLJ:
    """Explicit-lifetime reference bridge to the installed C++ LJ plugin.

    This is not metatomic.ExternalModel or a supported generic Python loader.
    Keep use in a context manager. The library owns registered plugin lifetime.
    """

    def __init__(self, options=None):
        from metatomic._c_api import mta_model_t
        from metatomic._c_lib import _get_library

        from metatensor import ExternalCpuArray, register_external_data_wrapper

        global _PLUGIN_LOADED
        self._library = _get_library()
        self._closed = True
        plugin = (
            Path(metatomic.__file__).parent / "libexec/metatomic/metatomic-lj-plugin.so"
        )
        if not plugin.is_file():
            raise FileNotFoundError(
                f"LJ plugin from the pinned local build is missing: {plugin}"
            )
        with _PLUGIN_LOCK:
            register_external_data_wrapper(
                "metatensor::SimpleDataArray", ExternalCpuArray
            )
            if not _PLUGIN_LOADED:
                self._library.mta_load_plugin(str(plugin).encode())
                _PLUGIN_LOADED = True
        self._model = mta_model_t()
        self._library.mta_load_model(
            b"metatomic-lj-model",
            json.dumps(DEFAULT_OPTIONS if options is None else options).encode(),
            b"metatomic-lj-plugin",
            ctypes.byref(self._model),
        )
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise ValueError("native reference model is closed")

    def _json_callback(self, name):
        from metatomic._c_api import mta_string_t
        from metatomic._status import check_status
        from metatomic._utils import _string_from_mta

        self._check_open()
        output = mta_string_t()
        check_status(getattr(self._model, name)(self._model.data, ctypes.byref(output)))
        return json.loads(_string_from_mta(output))

    def capabilities(self):
        return ModelCapabilities.from_dict(self._json_callback("capabilities"))

    def metadata(self):
        return ModelMetadata.from_dict(self._json_callback("metadata"))

    def requested_pair_lists(self):
        return [
            PairListOptions.from_dict(p)
            for p in self._json_callback("requested_pair_lists")
        ]

    def requested_inputs(self):
        return [Quantity.from_dict(q) for q in self._json_callback("requested_inputs")]

    def execute(
        self, systems, selected_atoms, requested_outputs, check_consistency=True
    ):
        from metatomic._c_api import mta_system_t, mts_tensormap_t

        self._check_open()
        pointers = (ctypes.POINTER(mta_system_t) * len(systems))(
            *[s.as_mta_system_t() for s in systems]
        )
        outputs = (ctypes.POINTER(mts_tensormap_t) * len(requested_outputs))()
        selected = None if selected_atoms is None else selected_atoms.as_mts_labels_t()
        # Keep all input owners alive throughout the synchronous call. The C
        # executor owns validation; execute_inner is not the engine entry point.
        self._library.mta_execute_model(
            self._model,
            pointers,
            len(systems),
            selected,
            json.dumps([q.to_dict() for q in requested_outputs]).encode(),
            check_consistency,
            outputs,
            len(outputs),
        )
        return [TensorMap.unsafe_from_ptr(p) for p in outputs]

    def close(self):
        from metatomic._status import check_status

        if not self._closed:
            self._closed = True
            if self._model.unload:
                check_status(self._model.unload(self._model.data))

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()


class ReferenceLJ:
    """CPU float64 teaching contract; no plugin registration or native callbacks.

    Systems use NumPy storage at the engine boundary. Backend subclasses only
    change the differentiable pair kernel. No public backend model base is implied.
    """

    def __init__(self, sigma=1.0, epsilon=1.0, cutoff=3.0):
        if not all(np.isfinite(x) and x > 0 for x in (sigma, epsilon, cutoff)):
            raise ValueError("sigma, epsilon and cutoff must be finite and positive")
        self.sigma, self.epsilon, self.cutoff = sigma, epsilon, cutoff

    def capabilities(self):
        return ModelCapabilities(
            atomic_types=[1],
            interaction_range=self.cutoff,
            length_unit="angstrom",
            supported_devices=["cpu"],
            dtype="float64",
            outputs=[
                Quantity(
                    name="energy",
                    unit="eV",
                    sample_kind="system",
                    gradients=["positions"],
                ),
                Quantity(name="energy", unit="eV", sample_kind="atom"),
            ],
        )

    def metadata(self):
        return ModelMetadata(
            name="Python LJ contract reference",
            description="Energy-shifted LJ; upstream selection and output conventions",
        )

    def requested_pair_lists(self):
        return [pair_options(self.cutoff)]

    def requested_inputs(self):
        return []

    def pair_energy_gradient(self, vectors):
        raise NotImplementedError

    def execute(
        self, systems, selected_atoms, requested_outputs, check_consistency=True
    ):
        # Limited explicit validation for this experiment. The eventual public
        # execution wrapper must delegate full checks/conversions to the core.
        if not check_consistency:
            raise ValueError("the reference only supports checked execution")
        if selected_atoms is not None:
            if selected_atoms.names != ["system", "atom"]:
                raise ValueError("selection labels must be system, atom")
            for system_i, atom_i in selected_atoms.values:
                if not 0 <= system_i < len(systems) or not 0 <= atom_i < len(
                    systems[system_i]
                ):
                    raise ValueError("selected atom is out of bounds")
        for request in requested_outputs:
            if (
                request.name != "energy"
                or request.unit != "eV"
                or request.sample_kind not in ("system", "atom")
            ):
                raise ValueError("unsupported energy request")
            if set(request.gradients) - {"positions"} or (
                request.sample_kind == "atom" and request.gradients
            ):
                raise ValueError("only system energy positions gradients are supported")
        results = []
        for request in requested_outputs:
            atom_rows = (
                [[s, a] for s, system in enumerate(systems) for a in range(len(system))]
                if selected_atoms is None
                else selected_atoms.values.tolist()
            )
            atom_lookup = {tuple(row): index for index, row in enumerate(atom_rows)}
            is_atomic = request.sample_kind == "atom"
            values = np.zeros((len(atom_rows) if is_atomic else len(systems), 1))
            gradients = [np.zeros((len(system), 3)) for system in systems]
            for system_i, system in enumerate(systems):
                if (
                    system.arrays_backend != "numpy"
                    or system.length_unit != "angstrom"
                    or system.positions.dtype != np.float64
                    or np.any(system.types != 1)
                ):
                    raise ValueError(
                        "reference requires NumPy float64 hydrogen Systems in angstrom"
                    )
                pairs = system.pairs(pair_options(self.cutoff))
                topology = pairs.samples.values
                first, second = topology[:, 0], topology[:, 1]
                vectors = (
                    system.positions[second]
                    - system.positions[first]
                    + topology[:, 2:] @ system.cell
                )
                if np.any(np.linalg.norm(vectors, axis=1) == 0):
                    raise ValueError("coincident pair endpoints")
                np.testing.assert_allclose(vectors, pairs.values[:, :, 0], atol=1e-12)
                energies, derivatives = self.pair_energy_gradient(vectors)
                for pair, (a, b) in enumerate(zip(first, second, strict=True)):
                    selected_a, selected_b = (
                        (system_i, int(a)) in atom_lookup,
                        (system_i, int(b)) in atom_lookup,
                    )
                    if is_atomic:
                        if selected_a:
                            values[atom_lookup[(system_i, int(a))], 0] += (
                                0.5 * energies[pair]
                            )
                        if selected_b:
                            values[atom_lookup[(system_i, int(b))], 0] += (
                                0.5 * energies[pair]
                            )
                    else:
                        scale = 0.5 * (int(selected_a) + int(selected_b))
                        values[system_i, 0] += scale * energies[pair]
                        gradients[system_i][a] -= scale * derivatives[pair]
                        gradients[system_i][b] += scale * derivatives[pair]
            samples = (
                labels(["system", "atom"], atom_rows)
                if is_atomic
                else labels(["system"], [[s] for s in range(len(systems))])
            )
            properties = labels(["energy"], [[0]])
            block = TensorBlock(values, samples, [], properties)
            if request.gradients:
                gradient_rows = [
                    [s, s, a]
                    for s, system in enumerate(systems)
                    for a in range(len(system))
                ]
                gradient_values = (
                    np.concatenate(gradients) if gradients else np.empty((0, 3))
                )
                gradient = TensorBlock(
                    gradient_values[:, :, None],
                    labels(["sample", "system", "atom"], gradient_rows),
                    [labels(["xyz"], [[0], [1], [2]])],
                    properties,
                )
                block.add_gradient("positions", gradient)
            results.append(TensorMap(labels(["_"], [[0]]), [block]))
        return results


class NumpyLJ(ReferenceLJ):
    def pair_energy_gradient(self, vectors):
        squared = np.sum(vectors * vectors, axis=1)
        active = squared < self.cutoff**2
        sixth = (self.sigma**2 / squared) ** 3
        cutoff_sixth = (self.sigma / self.cutoff) ** 6
        energy = 4 * self.epsilon * (sixth**2 - sixth - cutoff_sixth**2 + cutoff_sixth)
        gradient = (24 * self.epsilon * (sixth - 2 * sixth**2) / squared)[
            :, None
        ] * vectors
        return np.where(active, energy, 0), np.where(active[:, None], gradient, 0)


class TorchLJ(ReferenceLJ):
    def pair_energy_gradient(self, vectors):
        import torch

        vector = torch.tensor(vectors.copy(), dtype=torch.float64, requires_grad=True)
        squared = (vector * vector).sum(dim=1)
        sixth = (self.sigma**2 / squared) ** 3
        cutoff_sixth = (self.sigma / self.cutoff) ** 6
        shifted = 4 * self.epsilon * (sixth**2 - sixth - cutoff_sixth**2 + cutoff_sixth)
        energy = torch.where(squared < self.cutoff**2, shifted, 0.0)
        derivative = torch.autograd.grad(energy.sum(), vector)[0]
        return energy.detach().numpy().copy(), derivative.detach().numpy().copy()


class JaxLJ(ReferenceLJ):
    def __init__(self, sigma=1.0, epsilon=1.0, cutoff=3.0):
        import jax
        import jax.numpy as jnp

        super().__init__(sigma, epsilon, cutoff)
        if not jax.config.x64_enabled:
            raise ValueError("enable jax_enable_x64 before constructing JaxLJ")

        def energy(vector, parameters):
            sigma, epsilon, cutoff = parameters
            squared = jnp.sum(vector * vector)
            sixth = (sigma**2 / squared) ** 3
            cutoff_sixth = (sigma / cutoff) ** 6
            shifted = 4 * epsilon * (sixth**2 - sixth - cutoff_sixth**2 + cutoff_sixth)
            return jnp.where(squared < cutoff**2, shifted, 0.0)

        self._kernel = jax.jit(jax.vmap(jax.value_and_grad(energy), in_axes=(0, None)))

    def pair_energy_gradient(self, vectors):
        values, gradient = self._kernel(
            vectors, np.array([self.sigma, self.epsilon, self.cutoff])
        )
        return np.array(values), np.array(gradient)
