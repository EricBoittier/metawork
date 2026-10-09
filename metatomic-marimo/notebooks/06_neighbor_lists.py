import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 06 · From a real neighbor search to differentiable pair energy

    **Simulation · Executable tutorial**

    **Learning objective.** Build periodic pair topology and reconstruct vectors inside a differentiable kernel.

    **Prerequisites.** 01 and 03.


    The [published neighbor-list tutorial](https://docs.metatensor.org/metatomic/latest/examples/3-atomistic-model-with-nl.html)
    assigns neighbor-list construction to the engine. We retain that separation and
    use the new core names: `PairListOptions`, `add_pairs`, and `pairs`.
    This is an original small example, with a shifted-force Lennard–Jones potential
    in reduced numerical parameters (sigma=1 Å, epsilon=1 eV). It is not an argon fit.

    **Expected result.** The three-atom fixture has four directed pairs, including a periodic image; full-list energy carries a factor of one half.

    ## Build a periodic geometry and discover its pairs
    ASE returns both directions of each pair. We keep the **full list**, including
    periodic image shifts. The atom at x=4.0 interacts with x=0.2 across the boundary.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import jax
    import jax.numpy as jnp
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    from metatomic import System, PairListOptions
    from metatensor import Labels, TensorBlock
    from metatomic_marimo.visualization import structure_panel

    mo.show_code(jax.config.update("jax_enable_x64", True), position="above")
    return (
        Atoms,
        Labels,
        PairListOptions,
        System,
        TensorBlock,
        jax,
        jnp,
        mo,
        neighbor_list,
        np,
        structure_panel,
    )


@app.cell
def _(Atoms, mo, neighbor_list, np):
    atoms = Atoms(
        "H3",
        positions=[[0.2, 1.0, 1.0], [1.4, 1.0, 1.0], [4.0, 1.0, 1.0]],
        cell=5 * np.eye(3),
        pbc=True,
    )
    cutoff = 2.0
    first, second, shifts, vectors = neighbor_list(
        "ijSD", atoms, cutoff, self_interaction=False
    )
    first, second, shifts = (
        first.astype(np.int32),
        second.astype(np.int32),
        shifts.astype(np.int32),
    )
    assert np.allclose(
        vectors,
        atoms.positions[second] - atoms.positions[first] + shifts @ atoms.cell.array,
    )
    _display = mo.ui.table(
        [
            {
                "i": int(i),
                "j": int(j),
                "shift": s.tolist(),
                "distance": float(np.linalg.norm(d)),
            }
            for i, j, s, d in zip(first, second, shifts, vectors, strict=True)
        ],
        selection=None,
    )
    mo.show_code(_display, position="above")
    return atoms, cutoff, first, second, shifts, vectors


@app.cell
def _(mo):
    mo.md(r"""
    ## Attach the full list before borrowing System arrays
    Samples identify `(first_atom, second_atom, cell_shift_a, cell_shift_b, cell_shift_c)`.
    Values are displacement vectors with shape `(n_pairs, 3, 1)`. Ownership transfers
    to `System`; retrieve the block through `pairs(options)` after attaching it.
    """)
    return


@app.cell
def _(
    Labels,
    PairListOptions,
    System,
    TensorBlock,
    atoms,
    cutoff,
    first,
    mo,
    np,
    second,
    shifts,
    vectors,
):
    system = System(
        "angstrom",
        atoms.numbers.astype(np.int32),
        atoms.positions.copy(),
        atoms.cell.array.copy(),
        atoms.pbc.copy(),
    )
    options = PairListOptions(cutoff=cutoff, full_list=True)
    pairs = TensorBlock(
        values=vectors[:, :, None].copy(),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            np.column_stack([first, second, shifts]).astype(np.int32),
        ),
        components=[Labels("xyz", np.arange(3, dtype=np.int32).reshape(-1, 1))],
        properties=Labels("distance", np.array([[0]], dtype=np.int32)),
    )
    system.add_pairs(options, pairs)
    _display = mo.md(f"Attached {len(system.pairs(options).samples)} directed pairs.")
    mo.show_code(_display, position="above")
    return (system,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Rebuild displacement vectors inside the differentiated function
    A core System with attached pairs cannot currently enter JAX transformations.
    Pass the integer topology separately and compute vectors from traced positions
    and cell. Using cached vectors as constants would silently lose geometry derivatives.

    Each physical pair occurs twice in this full list, so the sum has a factor 1/2.
    Subtracting the value and slope at the cutoff makes energy and force go to zero
    there. Neighbor topology is fixed during a derivative; rebuild it after geometry
    changes. Production engines typically use a skin and a rebuild criterion.
    """)
    return


@app.cell
def _(
    atoms,
    cutoff,
    first,
    jax,
    jnp,
    mo,
    np,
    second,
    shifts,
    structure_panel,
    system,
):
    def pair_energy(positions, cell, first, second, shifts, cutoff):
        d = positions[second] - positions[first] + shifts @ cell
        r = jnp.linalg.norm(d, axis=1)

        def potential(x):
            return 4.0 * (x**-12 - x**-6)

        slope_at_cutoff = 24.0 * (cutoff**-7 - 2.0 * cutoff**-13)
        shifted = potential(r) - potential(cutoff) - (r - cutoff) * slope_at_cutoff
        return 0.5 * jnp.sum(jnp.where(r < cutoff, shifted, 0.0))

    pair_evaluate = jax.jit(jax.value_and_grad(pair_energy, argnums=(0, 1)))
    energy, (position_gradient, cell_gradient) = pair_evaluate(
        jnp.array(atoms.positions),
        jnp.array(atoms.cell.array),
        jnp.array(first),
        jnp.array(second),
        jnp.array(shifts),
        cutoff,
    )
    forces = -np.asarray(position_gradient)
    strain_gradient = atoms.positions.T @ np.asarray(
        position_gradient
    ) + atoms.cell.array.T @ np.asarray(cell_gradient)
    stress = (strain_gradient + strain_gradient.T) / (2 * atoms.get_volume())
    assert np.allclose(forces.sum(axis=0), 0, atol=1e-12)
    _display = structure_panel(
        system, energy=energy, forces=forces, stress=stress, force_scale=0.15
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Try it
    Move the last atom from x=4.0 to x=3.8 and rerun the neighbor search. Which image
    shift connects it to the first atom? Remove the factor 1/2 and predict the energy
    and force error before running. For a half list, keep one representative of each
    periodic pair and remove that factor; selecting only `i < j` is not a general
    solution when self-images are present.

    **Roadmap connection:** this covers the pair-handling part of a custom model and
    engine tutorial. A future model base class must declare these requests, and a
    production engine must satisfy them. We do not invent the pending request method.

    **Worked answer.** Removing 1/2 doubles energy, forces and stress. Moving an atom requires rebuilding topology when the neighborhood changes; a frozen pair list is valid only within its declared update policy.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Compare with the accepted C++ LJ reference
    This chapter deliberately uses a **force-shifted full-list** teaching potential. The upstream C++ LJ plugin uses an **energy-shifted half-list** potential and currently exposes no strain gradient. For exact C++/NumPy/Torch/JAX parity, use [chapter 21](/notebooks/cpp-lj/); do not compare these two conventions numerically as if they were the same model.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce and continue
    Run `uv run marimo edit notebooks/06_neighbor_lists.py` from this project. Use the default controls to reproduce the reference result, then vary one input at a time. Replace `.value` controls with numeric values when copying to a script. Computation cells show their executed source; plotting helpers belong to this cookbook.

    This edition targets core System revision `b02b9ff3`, Python 3.12 and CPU float64. Source and dependencies are pinned in the guide; HTML exports preserve results but cannot rerun Python controls.

    [Published neighbor-list tutorial](https://docs.metatensor.org/metatomic/latest/examples/3-atomistic-model-with-nl.html) · [Guide](/guide) · [API status](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
