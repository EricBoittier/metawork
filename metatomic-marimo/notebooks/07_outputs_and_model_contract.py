import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 07 · From numerical predictions to labeled outputs

    **Contracts · Executable tutorial**

    **Learning objective.** Encode energies and their derivatives with correct labels, signs and units, then recover them after serialization.

    **Prerequisites.** 01–04.


    The [published export tutorial](https://docs.metatensor.org/metatomic/latest/examples/1-export-atomistic-model.html)
    uses the established TorchScript `AtomisticModel` wrapper. Here we prepare the
    **data side** of the future core model interface using APIs present in this branch.
    Saving a TensorMap saves predictions, not an executable model.

    **Expected result.** The saved prediction returns E = 0.2 eV, ±2 eV/Å forces and σxx = 0.0192 eV/Å³.

    ## Specify what the model produces
    Core `Quantity` is a named descriptor, and core capabilities take a **list** of
    these descriptors. This differs from the TorchScript output dictionary.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import json
    import tempfile
    from pathlib import Path
    import metatensor
    import metatomic
    from metatensor import Labels, TensorBlock, TensorMap
    from metatomic import Quantity, ModelCapabilities, ModelMetadata

    mo.show_code(position="above")
    return (
        Labels,
        ModelCapabilities,
        ModelMetadata,
        Path,
        Quantity,
        TensorBlock,
        TensorMap,
        json,
        metatensor,
        metatomic,
        mo,
        np,
        tempfile,
    )


@app.cell
def _(ModelCapabilities, ModelMetadata, Quantity, json, mo):
    request = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    capabilities = ModelCapabilities(
        outputs=[request],
        atomic_types=[1],
        interaction_range=2.0,
        length_unit="angstrom",
        supported_devices=["cpu"],
        dtype="float64",
    )
    metadata = ModelMetadata(
        name="harmonic-data-example",
        description="Labeled predictions for a teaching model",
    )
    manifest = {"metadata": metadata.to_dict(), "capabilities": capabilities.to_dict()}
    manifest_json = json.dumps(manifest, indent=2)
    assert (
        ModelCapabilities.from_dict(json.loads(manifest_json)["capabilities"])
        == capabilities
    )
    mo.show_code(mo.json(manifest), position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Label the energy and its derivatives
    Use the known two-atom result at r=1.2 Å: E=0.2 eV, forces ±2 eV/Å, volume
    125 Å³, and xx stress 0.0192 eV/Å³. This isolates the output format from the
    energy implementation.

    - Energy values: `(systems, properties)`; samples contain `system`.
    - Position gradients: `(gradient_samples, xyz, properties)`; samples contain
      `sample` (row in the energy block), `system`, and `atom`.
    - Strain gradients: `(gradient_samples, xyz_1, xyz_2, properties)`; samples contain `sample`.

    Store **dE/dR = −forces**, and **dE/dstrain = volume × stress**. Storing forces
    or stress directly under these gradient names would have the wrong sign or units.
    """)
    return


@app.cell
def _(Labels, TensorBlock, TensorMap, mo, np):
    energy_value = 0.2
    force_values = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    stress_values = np.diag([0.0192, 0.0, 0.0])
    volume = 125.0
    properties = Labels("energy", np.array([[0]], dtype=np.int32))
    energy_block = TensorBlock(
        values=np.array([[energy_value]]),
        samples=Labels("system", np.array([[0]], dtype=np.int32)),
        components=[],
        properties=properties,
    )
    energy_block.add_gradient(
        "positions",
        TensorBlock(
            values=-force_values[:, :, None],
            samples=Labels(
                ["sample", "system", "atom"],
                np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int32),
            ),
            components=[Labels("xyz", np.arange(3, dtype=np.int32).reshape(-1, 1))],
            properties=properties,
        ),
    )
    energy_block.add_gradient(
        "strain",
        TensorBlock(
            values=(volume * stress_values)[None, :, :, None],
            samples=Labels("sample", np.array([[0]], dtype=np.int32)),
            components=[
                Labels("xyz_1", np.arange(3, dtype=np.int32).reshape(-1, 1)),
                Labels("xyz_2", np.arange(3, dtype=np.int32).reshape(-1, 1)),
            ],
            properties=properties,
        ),
    )
    energy_map = TensorMap(Labels("_", np.array([[0]], dtype=np.int32)), [energy_block])
    _display = mo.md(
        f"Energy shape: `{energy_map.block().values.shape}`; gradients: `{energy_map.block().gradients_list()}`."
    )
    mo.show_code(_display, position="above")
    return energy_map, force_values, stress_values, volume


@app.cell
def _(mo):
    mo.md(r"""
    ## Save predictions and recover forces/stress
    The temporary file makes this notebook repeatable without overwriting your data.
    For a persistent output, replace the temporary directory with your own path.
    JSON preserves descriptors; `.mts` preserves labeled tensor data. Neither file
    contains a Python plugin, weights, or an executable model.
    """)
    return


@app.cell
def _(
    Path,
    energy_map,
    force_values,
    metatensor,
    mo,
    np,
    stress_values,
    tempfile,
    volume,
):
    with tempfile.TemporaryDirectory() as directory:
        output_path = Path(directory) / "energy.mts"
        metatensor.save(str(output_path), energy_map)
        reloaded = metatensor.load(str(output_path))
    recovered_forces = -reloaded.block().gradient("positions").values[:, :, 0]
    recovered_stress = reloaded.block().gradient("strain").values[0, :, :, 0] / volume
    np.testing.assert_allclose(recovered_forces, force_values)
    np.testing.assert_allclose(recovered_stress, stress_values)
    _display = mo.vstack(
        [
            mo.md(
                "✓ Saved prediction data round-trips with the correct force sign and stress normalization."
            ),
            recovered_forces,
            recovered_stress,
        ]
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check the actual API before writing integration code
    The next check inspects the imported core package. An absent symbol is a concrete
    blocker, not a suggestion to substitute a similarly named TorchScript type.
    When these interfaces land, replace the teaching classes in notebooks 02–04 with
    the documented subclasses, then add model save/load/evaluate tests.
    """)
    return


@app.cell
def _(metatomic, mo):
    api_status = [
        {"symbol": name, "available in core": hasattr(metatomic, name)}
        for name in [
            "System",
            "Quantity",
            "ModelCapabilities",
            "ModelMetadata",
            "Model",
            "TorchModel",
            "NumpyModel",
            "JaxModel",
            "load_model",
        ]
    ]
    mo.show_code(mo.ui.table(api_status, selection=None), position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Roadmap: what is still missing?
    The core package currently has no public Python model execution/loader or generic
    model subclasses. Public native execution, Python plugin registration, and
    executable export remain upstream work; chapter 21 demonstrates a private bridge. The existing `metatomic.torch` exporter
    is a separate working route described by the linked tutorial; this notebook does
    not claim binary or graph compatibility with it.

    **Try it:** add a second system sample. Its gradient `sample` index must refer to
    the second energy row. Give the two cells different volumes and normalize each
    strain gradient by its own volume. Do not copy a single volume across a batch.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    A selected system has ID 4 but occupies energy row 1. Which index belongs in its gradient’s `sample` column?

    **Worked answer.** For a second system in energy row 1, gradient sample indices refer to row 1. Normalize that system’s strain gradient by its own volume. The sample index is an output-row index, not necessarily a system ID after selection.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce and continue
    Run `uv run marimo edit notebooks/07_outputs_and_model_contract.py` from this project. Use the default controls to reproduce the reference result, then vary one input at a time. Replace `.value` controls with numeric values when copying to a script. Computation cells show their executed source; plotting helpers belong to this cookbook.

    This edition targets core System revision `b02b9ff3`, Python 3.12 and CPU float64. Source and dependencies are pinned in the guide; HTML exports preserve results but cannot rerun Python controls.

    [Core energy contract](https://github.com/metatensor/metatomic/blob/12b24b14/metatomic-core/src/quantity/checks.rs) · [Guide](/guide) · [API status](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
