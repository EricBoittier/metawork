import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 01 · Describing outputs and attaching pair lists

    **Foundations · Executable tutorial**

    **Learning objective.** Describe model outputs and attach a labeled pair list without confusing metadata with execution.

    **Prerequisites.** 00 — System.

    These are implemented core APIs. Metadata and capabilities describe a model;
    they do not register it, load a plugin, or execute it.

    Energy derivatives named `positions` and `strain` are derivatives of energy.
    Forces negate position derivatives; stress divides strain derivatives by volume.

    **Expected result.** One half-list pair has displacement (1.2, 0, 0) Å; its values have shape (1, 3, 1).
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from metatomic import (
        System,
        Quantity,
        ModelCapabilities,
        ModelMetadata,
        PairListOptions,
    )
    from metatensor import Labels, TensorBlock, TensorMap

    mo.show_code(position="above")
    return (
        Labels,
        ModelCapabilities,
        ModelMetadata,
        PairListOptions,
        Quantity,
        System,
        TensorBlock,
        TensorMap,
        mo,
        np,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Describe model outputs
    Construct a named `Quantity`, put it in `ModelCapabilities.outputs`, and attach human-readable metadata. This describes behavior; it does not execute a model.
    """)
    return


@app.cell
def _(ModelCapabilities, ModelMetadata, Quantity, mo):
    energy_request = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    capabilities = ModelCapabilities(
        outputs=[energy_request],
        atomic_types=[1],
        interaction_range=2.0,
        length_unit="angstrom",
        supported_devices=["cpu"],
        dtype="float64",
    )
    metadata = ModelMetadata(
        name="Tutorial pair potential",
        description="Teaching example, not a fitted hydrogen model",
    )
    assert ModelCapabilities.from_dict(capabilities.to_dict()) == capabilities
    _display = mo.accordion(
        {
            "Energy request": mo.json(energy_request.to_dict()),
            "Capabilities": mo.json(capabilities.to_dict()),
            "Metadata": mo.json(metadata.to_dict()),
        }
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A hand-built half pair list
    For this two-atom structure we know the single pair `(0, 1)` in advance.
    A production engine must build/rebuild its neighbor list using the requested
    cutoff, PBC, and full/half-list convention. The values below are displacement
    vectors, not scalar distances. This example does not implement a neighbor search.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Attach known pair vectors and custom data
    This hand-built pair is useful for learning storage. Notebook 06 replaces it with a real periodic neighbor search.
    """)
    return


@app.cell
def _(Labels, PairListOptions, System, TensorBlock, TensorMap, mo, np):
    pair_system = System(
        "angstrom",
        np.array([1, 1], dtype=np.int32),
        np.array([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]]),
        5.0 * np.eye(3),
        np.ones(3, dtype=np.bool_),
    )
    pair_options = PairListOptions(cutoff=2.0, full_list=False, requestors=["tutorial"])
    pair_block = TensorBlock(
        values=np.array([[[1.2], [0.0], [0.0]]]),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            np.array([[0, 1, 0, 0, 0]], dtype=np.int32),
        ),
        components=[Labels("xyz", np.arange(3, dtype=np.int32).reshape(-1, 1))],
        properties=Labels("distance", np.array([[0]], dtype=np.int32)),
    )
    pair_system.add_pairs(pair_options, pair_block)
    # pair_block has transferred ownership: access through pair_system from now on.
    custom = TensorMap(
        Labels("_", np.array([[0]], dtype=np.int32)),
        [
            TensorBlock(
                values=np.array([[42.0]]),
                samples=Labels("system", np.array([[0]], dtype=np.int32)),
                components=[],
                properties=Labels("feature", np.array([[0]], dtype=np.int32)),
            )
        ],
    )
    pair_system.add_custom_data("tutorial::feature", custom)
    _display = mo.md(
        f"Attached **{len(pair_system.known_pairs())}** pair list and custom data `{pair_system.known_custom_data()}`."
    )
    mo.show_code(_display, position="above")
    return pair_options, pair_system


@app.cell
def _(mo):
    mo.md(r"""
    ## Retrieve data through System
    The original blocks transferred ownership. Read the stored data through the System instead of reusing the transferred objects.
    """)
    return


@app.cell
def _(mo, pair_options, pair_system):
    _display = mo.vstack(
        [
            mo.md("Stored displacement (shape: pairs × xyz × properties):"),
            pair_system.pairs(pair_options).values,
            mo.md("Stored custom feature:"),
            pair_system.custom_data("tutorial::feature").block().values,
        ]
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Current JAX restriction:** this enriched system cannot be supplied to `jax.jit`
    or `jax.grad`. Keep discrete pair indices/image shifts as separate model inputs
    and reconstruct displacement vectors from traced positions and cell. A cached
    displacement array is not automatically differentiable with respect to geometry.
    """)
    return


@app.cell
def _(mo):
    from metatomic_marimo.visualization import structure_panel

    mo.show_code(position="above")
    return (structure_panel,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Inspect the structure
    The pair vector should point from atom 0 to atom 1. Displacement vectors must be rebuilt from differentiable geometry when calculating forces.
    """)
    return


@app.cell
def _(mo, pair_system, structure_panel):
    mo.show_code(structure_panel(pair_system), position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise
    Change the cutoff in PairListOptions after attaching a list. Does the new request still identify the stored list? Then inspect the exact sample labels in notebook 06.


    **Worked answer.** A list is identified by its options, not only its displacement values. A different cutoff requests a different list; querying it requires an appropriate attached list.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce and continue
    Run `uv run marimo edit notebooks/01_metadata_and_pairs.py` from this project. Use the default controls to reproduce the reference result, then vary one input at a time. Replace `.value` controls with numeric values when copying to a script. Computation cells show their executed source; plotting helpers belong to this cookbook.

    This edition targets core System revision `b02b9ff3`, Python 3.12 and CPU float64. Source and dependencies are pinned in the guide; HTML exports preserve results but cannot rerun Python controls.

    [Core quantity validation](https://github.com/metatensor/metatomic/blob/12b24b14/metatomic-core/src/quantity/checks.rs) · [Guide](/guide) · [API status](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
