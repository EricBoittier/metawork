"""Execute notebooks and check physics independently of their implementation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def notebooks():
    results = {}
    for path in sorted((ROOT / "notebooks").glob("*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, definitions = module.app.run()
        results[path.stem] = definitions
    return results


def test_all_notebooks_execute(notebooks):
    assert len(notebooks) == 22
    pairs = notebooks["01_metadata_and_pairs"]
    assert pairs["pair_system"].known_custom_data() == ["tutorial::feature"]
    np.testing.assert_allclose(
        pairs["pair_system"].pairs(pairs["pair_options"]).values.ravel(), [1.2, 0, 0]
    )


def test_analytic_values_and_backend_parity(notebooks):
    for name in (
        "02_numpy_model",
        "03_jax_energy_forces_stress",
        "04_torch_energy_forces_stress",
    ):
        defs = notebooks[name]
        np.testing.assert_allclose(defs["energy"], 0.2, atol=1e-12)
        np.testing.assert_allclose(
            defs["forces"], [[2.0, 0, 0], [-2.0, 0, 0]], atol=1e-12
        )
        np.testing.assert_allclose(defs["stress"], np.diag([0.0192, 0, 0]), atol=1e-12)
    torch_defs = notebooks["04_torch_energy_forces_stress"]
    for actual, expected in zip(
        torch_defs["compiled_results"],
        (torch_defs["energy"], torch_defs["forces"], torch_defs["stress"]),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def make_system(positions, cell, backend):
    from metatomic import System

    if backend == "jax":
        import jax.numpy as xp

        return System(
            "angstrom",
            xp.array([1, 1], dtype=xp.int32),
            xp.array(positions),
            xp.array(cell),
            xp.ones(3, dtype=xp.bool_),
            arrays_backend="jax",
        )
    if backend == "torch":
        import torch

        return System(
            "angstrom",
            torch.tensor([1, 1], dtype=torch.int32),
            torch.tensor(positions, dtype=torch.float64),
            torch.tensor(cell, dtype=torch.float64),
            torch.ones(3, dtype=torch.bool),
            arrays_backend="torch",
        )
    return System(
        "angstrom",
        np.array([1, 1], dtype=np.int32),
        np.array(positions),
        np.array(cell),
        np.ones(3, dtype=np.bool_),
    )


def reference_energy(positions, cell):
    """Fixed orthogonal-box image, also valid for the small test deformations."""
    displacement = positions[1] - positions[0]
    image = np.round(np.linalg.solve(cell.T, displacement))
    displacement -= image @ cell
    return 5.0 * (np.linalg.norm(displacement) - 1.0) ** 2


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("cross_boundary", [False, True])
def test_forces_and_full_stress_finite_differences(notebooks, backend, cross_boundary):
    # Off-axis geometry checks shear components. Crossing a cell boundary makes
    # the explicit cell derivative nonzero, detecting a missing cell term.
    positions = np.array([[0.2, 1.1, 0.7], [1.4, 1.5, 0.9]])
    if cross_boundary:
        positions[1, 0] = 4.0
    cell = 5.0 * np.eye(3)
    if backend == "numpy":
        evaluate = notebooks["02_numpy_model"]["numpy_model"].evaluate
    else:
        name = (
            "03_jax_energy_forces_stress"
            if backend == "jax"
            else "04_torch_energy_forces_stress"
        )
        evaluate = notebooks[name]["evaluate"]
    energy, forces, stress = evaluate(make_system(positions, cell, backend))
    np.testing.assert_allclose(energy, reference_energy(positions, cell), atol=1e-12)
    epsilon = 1e-5
    expected_forces = np.zeros((2, 3))
    expected_stress = np.zeros((3, 3))
    for atom in range(2):
        for axis in range(3):
            shift = np.zeros((2, 3))
            shift[atom, axis] = epsilon
            expected_forces[atom, axis] = -(
                reference_energy(positions + shift, cell)
                - reference_energy(positions - shift, cell)
            ) / (2 * epsilon)
    for i in range(3):
        for j in range(3):
            strain = np.zeros((3, 3))
            strain[i, j] += epsilon / 2
            strain[j, i] += epsilon / 2
            plus, minus = np.eye(3) + strain, np.eye(3) - strain
            expected_stress[i, j] = (
                reference_energy(positions @ plus, cell @ plus)
                - reference_energy(positions @ minus, cell @ minus)
            ) / (2 * epsilon * abs(np.linalg.det(cell)))
    np.testing.assert_allclose(forces, expected_forces, atol=2e-8)
    np.testing.assert_allclose(stress, expected_stress, atol=2e-9)
    np.testing.assert_allclose(np.asarray(forces).sum(axis=0), 0, atol=1e-12)


def test_relaxation(notebooks):
    defs = notebooks["05_numpy_engine"]
    assert defs["converged"]
    energies = np.array([row["energy_eV"] for row in defs["history"]])
    assert np.all(np.diff(energies) <= 1e-14)
    assert defs["history"][-1]["max_force_eV_per_A"] < 1e-6
    initial = np.asarray(defs["initial"].positions)
    final = np.asarray(defs["relaxed"].positions)
    np.testing.assert_allclose(initial.mean(axis=0), final.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(final[1] - final[0]), 1.0, atol=1e-7)


def test_visualization_data_preserves_units_and_results(notebooks):
    import chemiscope

    from metatomic_marimo.visualization import make_dataset

    for name in (
        "02_numpy_model",
        "03_jax_energy_forces_stress",
        "04_torch_energy_forces_stress",
    ):
        defs = notebooks[name]
        dataset = make_dataset(
            [defs["system"]],
            energies=[defs["energy"]],
            forces=[defs["forces"]],
            stresses=[defs["stress"]],
        )
        payload = chemiscope.create_input(**dataset)
        assert payload["structures"][0]["size"] == 2
        np.testing.assert_allclose(
            payload["structures"][0]["cell"], (5 * np.eye(3)).ravel()
        )
        assert payload["properties"]["energy"]["target"] == "structure"
        assert payload["properties"]["stress_xx"]["target"] == "structure"
        np.testing.assert_allclose(
            payload["properties"]["pressure"]["values"], [-0.0064]
        )
        np.testing.assert_allclose(
            payload["properties"]["force_magnitude"]["values"], [2, 2]
        )
        # Arrow scaling affects geometry only, never the stored force values.
        assert "forces" in payload["shapes"]
        dataset["structures"][0].positions[:] = 0
        assert not np.allclose(np.asarray(defs["system"].positions), 0)


def test_trajectory_view_has_every_relaxation_step(notebooks):
    from metatomic_marimo.visualization import make_dataset

    history = notebooks["05_numpy_engine"]["history"]
    dataset = make_dataset(
        [row["system"] for row in history],
        energies=[row["energy_eV"] for row in history],
        forces=[row["forces"] for row in history],
    )
    assert len(dataset["structures"]) == len(history)
    np.testing.assert_allclose(
        dataset["properties"]["energy"]["values"], [row["energy_eV"] for row in history]
    )
    assert dataset["properties"]["max_force"]["values"][-1] < 1e-6


def test_neighbor_list_derivatives_and_periodic_images(notebooks):
    import jax.numpy as jnp

    defs = notebooks["06_neighbor_lists"]
    atoms = defs["atoms"]
    r, c = atoms.positions.copy(), atoms.cell.array.copy()
    first, second, shifts = defs["first"], defs["second"], defs["shifts"]
    assert np.any(shifts != 0)
    assert len(first) == 4  # Two physical pairs, both directions.

    def energy(positions, cell):
        return float(
            defs["pair_energy"](
                jnp.array(positions),
                jnp.array(cell),
                jnp.array(first),
                jnp.array(second),
                jnp.array(shifts),
                defs["cutoff"],
            )
        )

    epsilon = 1e-5
    forces = np.zeros_like(r)
    for i in range(3):
        for j in range(3):
            delta = np.zeros_like(r)
            delta[i, j] = epsilon
            forces[i, j] = -(energy(r + delta, c) - energy(r - delta, c)) / (
                2 * epsilon
            )
    np.testing.assert_allclose(defs["forces"], forces, atol=1e-7)
    plus, minus = np.eye(3), np.eye(3)
    plus[0, 0] += epsilon
    minus[0, 0] -= epsilon
    stress_xx = (energy(r @ plus, c @ plus) - energy(r @ minus, c @ minus)) / (
        2 * epsilon * atoms.get_volume()
    )
    np.testing.assert_allclose(defs["stress"][0, 0], stress_xx, atol=1e-8)


def test_labeled_output_contract(notebooks):
    defs = notebooks["07_outputs_and_model_contract"]
    block = defs["reloaded"].block()
    assert block.samples.names == ["system"]
    assert block.gradient("positions").samples.names == ["sample", "system", "atom"]
    assert block.gradient("strain").samples.names == ["sample"]
    assert block.gradient("strain").values.shape == (1, 3, 3, 1)
    np.testing.assert_allclose(defs["recovered_forces"], [[2, 0, 0], [-2, 0, 0]])
    np.testing.assert_allclose(defs["recovered_stress"], np.diag([0.0192, 0, 0]))


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_ase_calculator_and_energy_conservation(notebooks, backend):
    from ase import Atoms

    defs = notebooks["08_ase_dynamics"]
    atoms = Atoms(
        "H2", positions=[[1, 1, 1], [2.2, 1, 1]], cell=5 * np.eye(3), pbc=True
    )
    atoms.calc = defs["TutorialCalculator"](backend=backend)
    np.testing.assert_allclose(atoms.get_potential_energy(), 0.2, atol=1e-12)
    np.testing.assert_allclose(atoms.get_forces(), [[2, 0, 0], [-2, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(atoms.get_stress(), [0.0192, 0, 0, 0, 0, 0], atol=1e-12)
    records = defs["run_dynamics"](backend)
    total = np.array([r["energy"] + r["kinetic"] for r in records])
    assert np.max(np.abs(total - total[0])) < 1e-4
    reference = defs["records"]
    np.testing.assert_allclose(
        records[-1]["system"].positions, reference[-1]["system"].positions, atol=1e-11
    )


def test_batched_energies_and_forces(notebooks):
    defs = notebooks["09_batching_and_profiling"]
    distances = defs["distances"]
    np.testing.assert_allclose(
        defs["batch_energies"], 5 * (distances - 1) ** 2, atol=1e-12
    )
    np.testing.assert_allclose(
        -defs["batch_gradients"][:, 0, 0], 10 * (distances - 1), atol=1e-12
    )
    assert "tutorial::batched_harmonic_energy_and_forces" in defs["profile_report"]


def test_all_calculation_cells_show_their_code():
    """Prevent the app view reverting to results-only teaching material."""
    import ast

    for path in (ROOT / "notebooks").glob("*.py"):
        tree = ast.parse(path.read_text())
        assert not any(
            isinstance(n, ast.Attribute) and n.attr == "_unparsable_cell"
            for n in ast.walk(tree)
        )
        for cell in tree.body:
            if not isinstance(cell, ast.FunctionDef):
                continue
            # A prose-only cell contains a markdown call and its final return.
            prose = (
                len(cell.body) == 2
                and isinstance(cell.body[0], ast.Expr)
                and isinstance(cell.body[0].value, ast.Call)
                and isinstance(cell.body[0].value.func, ast.Attribute)
                and cell.body[0].value.func.attr == "md"
            )
            if not prose:
                assert any(
                    isinstance(n, ast.Attribute) and n.attr == "show_code"
                    for n in ast.walk(cell)
                ), path.name


def test_progress_json_errors_and_manifest(notebooks):
    import json

    defs = notebooks["10_api_progress_and_prototypes"]
    assert len(defs["json_roundtrips"]) == 5
    assert all(
        defs["json_roundtrips"][name] == obj
        for name, obj in defs["json_objects"].items()
    )
    assert [row["exception"] for row in defs["error_examples"]] == [
        "ValueError",
        "ValueError",
        "MetatomicError",
    ]
    for backend in ("numpy", "torch", "jax"):
        manifest = json.loads(defs["manifest_text"])
        manifest["backend"] = backend
        model = defs["load_prototype"](json.dumps(manifest))
        np.testing.assert_allclose(model.evaluate(defs["demo_system"])["energy"], 0.2)
    for key, value in (
        ("version", 2),
        ("backend", "unknown"),
        ("parameters", {"k": -1, "r0": 1}),
    ):
        manifest = json.loads(defs["manifest_text"])
        manifest[key] = value
        with pytest.raises(ValueError):
            defs["load_prototype"](json.dumps(manifest))


def test_progress_prototypes_with_periodic_geometry(notebooks):
    defs = notebooks["10_api_progress_and_prototypes"]
    positions = np.array([[0.1, 1.0, 1.0], [3.9, 1.1, 1.2]])
    cell = np.eye(3) * 5
    system = make_system(positions, cell, "numpy")
    step = 1e-5
    for model in defs["prototype_models"].values():
        result = model.evaluate(system)
        numeric_forces = np.zeros_like(positions)
        for atom in range(2):
            for axis in range(3):
                plus, minus = positions.copy(), positions.copy()
                plus[atom, axis] += step
                minus[atom, axis] -= step
                numeric_forces[atom, axis] = -(
                    model.evaluate(make_system(plus, cell, "numpy"))["energy"]
                    - model.evaluate(make_system(minus, cell, "numpy"))["energy"]
                ) / (2 * step)
        np.testing.assert_allclose(result["forces"], numeric_forces, atol=1e-8)
        plus, minus = np.eye(3), np.eye(3)
        plus[0, 0] += step
        minus[0, 0] -= step
        derivative = (
            model.evaluate(make_system(positions @ plus, cell @ plus, "numpy"))[
                "energy"
            ]
            - model.evaluate(make_system(positions @ minus, cell @ minus, "numpy"))[
                "energy"
            ]
        ) / (2 * step * abs(np.linalg.det(cell)))
        np.testing.assert_allclose(result["stress"][0, 0], derivative, atol=1e-8)
        with pytest.raises(ValueError, match="unsupported requested output"):
            model.evaluate(system, outputs=("unknown",))
