import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 00 · Meet `metatomic.System`

    **Foundations · Executable tutorial**

    **Learning objective.** Construct a System with NumPy, Torch or JAX; explain what the object owns and what an energy model must supply.

    **Prerequisites.** Basic NumPy arrays.

    A `System` describes a structure; it does not itself calculate an energy.
    This notebook uses the **core Python API on the `python-wrap-system` branch**,
    not the separate `metatomic.torch.System` API.

    Atomic types are `int32`, positions have shape `(N, 3)`, cell vectors are rows
    of a `(3, 3)` array, and PBC flags have shape `(3,)`. Positions and cell share
    a floating-point dtype. A nonperiodic cell vector must be zero.
    The length unit is explicit; labeling data with a unit does not rescale it.

    **Expected result.** The default structure has 2 atoms, positions (2, 3), a 5 Å cubic cell, and full periodicity.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    import jax
    import jax.numpy as jnp
    from metatomic import System

    mo.show_code(jax.config.update("jax_enable_x64", True), position="above")
    return System, jnp, mo, np, torch


@app.cell
def _(mo):
    mo.md(r"""
    ## Choose a bond length
    The slider is a marimo convenience. In a plain script, replace `bond_length.value` below with `1.2`.
    """)
    return


@app.cell
def _(mo):
    bond_length = mo.ui.slider(
        0.8, 1.6, step=0.05, value=1.2, label="Bond length (angstrom)"
    )
    mo.show_code(bond_length, position="above")
    return (bond_length,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Construct a System
    Write the four arrays explicitly. Keep atomic types int32 and both geometric arrays float64. The cell uses row vectors.
    """)
    return


@app.cell
def _(System, bond_length, mo, np):
    positions = np.array(
        [[1.0, 1.0, 1.0], [1.0 + bond_length.value, 1.0, 1.0]], dtype=np.float64
    )
    system = System(
        "angstrom",
        np.array([1, 1], dtype=np.int32),
        positions,
        5.0 * np.eye(3),
        np.ones(3, dtype=np.bool_),
        arrays_backend="numpy",
    )
    _display = mo.md(
        f"**{system.size} atoms**, backend `{system.arrays_backend}`, unit `{system.length_unit}`"
    )
    mo.show_code(_display, position="above")
    return positions, system


@app.cell
def _(mo):
    mo.md(r"""
    ## Read the atomic data
    Pair each type with its position. The table is only a display; the array properties are the API you use in calculations.
    """)
    return


@app.cell
def _(mo, system):
    _display = mo.ui.table(
        [
            {
                "atom": int(i),
                "type": int(z),
                "x": float(r[0]),
                "y": float(r[1]),
                "z": float(r[2]),
            }
            for i, (z, r) in enumerate(zip(system.types, system.positions))
        ]
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Three backends, the same structure
    Use one backend for all four constructor arrays, or set `arrays_backend`
    explicitly. DLPack transports storage; it does **not** promise to transport an
    autograd graph. Treat borrowed arrays as read-only and copy them before edits.
    Construct new systems when changing geometry.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Repeat with Torch and JAX
    Compare constructors: the geometry is identical, while the array backend changes. Backend conversion does not create an energy model.
    """)
    return


@app.cell
def _(System, jnp, mo, positions, system, torch):
    torch_system = System(
        "angstrom",
        torch.tensor([1, 1], dtype=torch.int32),
        torch.tensor(positions, dtype=torch.float64),
        5.0 * torch.eye(3, dtype=torch.float64),
        torch.ones(3, dtype=torch.bool),
        arrays_backend="torch",
    )
    jax_system = System(
        "angstrom",
        jnp.array([1, 1], dtype=jnp.int32),
        jnp.array(positions),
        5.0 * jnp.eye(3),
        jnp.ones(3, dtype=jnp.bool_),
        arrays_backend="jax",
    )
    _display = mo.ui.table(
        [
            {
                "backend": s.arrays_backend,
                "array class": str(type(s.positions)),
                "atoms": s.size,
            }
            for s in (system, torch_system, jax_system)
        ]
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Lifetime and differentiation
    The returned arrays keep storage alive. Attach pair lists/custom data **before**
    borrowing arrays; attached blocks/maps transfer ownership and must not be reused.

    JAX treats positions and cell as pytree leaves, with types/PBC/units as static
    metadata. Attached pair lists and custom data currently prevent JAX flattening.
    For Torch, extract tensors once, clone them, and start a fresh autograd graph.
    Notebooks 03 and 04 make these two workflows explicit.
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
    ## Inspect the geometry
    Rotate the cell and expand the coordinate table. Changing the bond length rebuilds the System and updates this view.
    """)
    return


@app.cell
def _(mo, structure_panel, system):
    mo.show_code(structure_panel(system), position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise
    Create a nonperiodic two-atom molecule: set all PBC flags to false and all cell vectors to zero. Why is dividing a strain gradient by this cell volume invalid?


    **Worked answer.** A nonperiodic molecule uses zero cell vectors. Its energy and forces remain meaningful, but stress normalized by a zero volume is undefined.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce and continue
    Run `uv run marimo edit notebooks/00_system.py` from this project. Use the default controls to reproduce the reference result, then vary one input at a time. Replace `.value` controls with numeric values when copying to a script. Computation cells show their executed source; plotting helpers belong to this cookbook.

    This edition targets core System revision `b02b9ff3`, Python 3.12 and CPU float64. Source and dependencies are pinned in the guide; HTML exports preserve results but cannot rerun Python controls.

    [Core System source](https://github.com/metatensor/metatomic/blob/b02b9ff3/python/metatomic_core/src/metatomic/_system.py) · [Guide](/guide) · [API status](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
