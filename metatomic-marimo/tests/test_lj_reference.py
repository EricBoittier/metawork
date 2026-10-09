"""Compare notebook reference implementations to the installed upstream C++ plugin."""

import json

import jax
import numpy as np
import pytest
from metatomic import MetatomicError, Quantity

from metatomic_marimo.lj_reference import (
    DEFAULT_OPTIONS,
    JaxLJ,
    NativeLJ,
    NumpyLJ,
    TorchLJ,
    labels,
    make_pair_system,
)

jax.config.update("jax_enable_x64", True)


def requests():
    return [
        Quantity(
            name="energy", unit="eV", sample_kind="system", gradients=["positions"]
        ),
        Quantity(name="energy", unit="eV", sample_kind="atom"),
    ]


def assert_outputs_equal(actual, expected):
    assert len(actual) == len(expected)
    for tensor, reference in zip(actual, expected, strict=True):
        assert tensor.keys == reference.keys
        block, reference_block = tensor.block(), reference.block()
        assert block.samples == reference_block.samples
        assert block.properties == reference_block.properties
        assert block.components == reference_block.components
        np.testing.assert_allclose(block.values, reference_block.values, atol=1e-11)
        assert block.gradients_list() == reference_block.gradients_list()
        for name in block.gradients_list():
            grad, ref_grad = block.gradient(name), reference_block.gradient(name)
            assert grad.samples == ref_grad.samples
            assert grad.components == ref_grad.components
            assert grad.properties == ref_grad.properties
            np.testing.assert_allclose(grad.values, ref_grad.values, atol=1e-10)


@pytest.mark.parametrize("model_type", [NumpyLJ, TorchLJ, JaxLJ])
@pytest.mark.parametrize("selection", [None, [[0, 0], [1, 1], [2, 0], [3, 1]]])
def test_cpp_contract(model_type, selection):
    systems = [
        make_pair_system(2.0),
        make_pair_system(1.2, transverse=0.2, image=True),
        make_pair_system(3.0),
        make_pair_system(3.2),
    ]
    selected = None if selection is None else labels(["system", "atom"], selection)
    with NativeLJ() as native:
        reference = native.execute(systems, selected, requests())
        assert native.requested_pair_lists() == model_type().requested_pair_lists()
        assert native.capabilities() == model_type().capabilities()
        assert native.requested_inputs() == []
    # Native output owners must remain valid after the model is closed.
    assert_outputs_equal(model_type().execute(systems, selected, requests()), reference)


@pytest.mark.parametrize("model_type", [NumpyLJ, TorchLJ, JaxLJ])
def test_force_finite_difference(model_type):
    model = model_type()
    system = make_pair_system(1.2, transverse=0.2, image=True)
    result = (
        model.execute([system], None, requests())[0]
        .block()
        .gradient("positions")
        .values
    )
    step = 1e-6
    plus = (
        model.execute(
            [make_pair_system(1.2 + step, transverse=0.2, image=True)], None, requests()
        )[0]
        .block()
        .values[0, 0]
    )
    minus = (
        model.execute(
            [make_pair_system(1.2 - step, transverse=0.2, image=True)], None, requests()
        )[0]
        .block()
        .values[0, 0]
    )
    np.testing.assert_allclose(result[1, 0, 0], (plus - minus) / (2 * step), atol=1e-8)
    np.testing.assert_allclose(result[0], -result[1], atol=1e-12)
    assert model.execute([system], None, []) == []
    bad = Quantity(name="energy", unit="eV", sample_kind="system", gradients=["strain"])
    with pytest.raises(ValueError, match="only system energy positions"):
        model.execute([system], None, [bad])


def test_native_execution_errors_and_lifetime():
    with NativeLJ() as native:
        assert native.execute([make_pair_system()], None, []) == []
        bad = Quantity(
            name="energy", unit="eV", sample_kind="system", gradients=["strain"]
        )
        with pytest.raises(MetatomicError):
            native.execute([make_pair_system()], None, [bad])
        # A failed request must not destroy the borrowed System or loaded model.
        system = make_pair_system()
        result = native.execute([system], None, requests())
        assert len(system) == 2
        assert result[0].block().values.shape == (1, 1)
    native.close()
    with pytest.raises(ValueError, match="closed"):
        native.execute([], None, [])
    options = json.loads(json.dumps(DEFAULT_OPTIONS))
    options["sigma"] = 1.0
    with pytest.raises(MetatomicError, match="non-string"):
        NativeLJ(options)


def test_upstream_selection_gap_is_reported():
    # Current LJ callback creates system samples for every input system, while
    # the checked executor expects only systems represented in the selection.
    with NativeLJ() as native:
        for selection in ([], [[0, 0]]):
            with pytest.raises(MetatomicError, match="invalid samples"):
                native.execute(
                    [make_pair_system(), make_pair_system(2.0)],
                    labels(["system", "atom"], selection),
                    requests(),
                )
