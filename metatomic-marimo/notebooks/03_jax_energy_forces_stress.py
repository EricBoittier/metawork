import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 03 · JAX: differentiate a `System`

    **Differentiation · Executable tutorial**

    **Learning objective.** Differentiate a core System pytree and verify stress through an independent homogeneous deformation.

    **Prerequisites.** 02; basic JAX grad/jit.

    This uses the branch's pytree registration: positions and cell are differentiable
    leaves; types, PBC, backend, and length unit are static metadata. Changing static
    metadata can trigger recompilation. No `JaxModel` base class is needed for this
    plain scalar energy function.

    As in notebook 02, use a two-atom periodic orthogonal box and a toy harmonic bond.
    The minimum-image branch must stay fixed locally during differentiation. There
    are derivative discontinuities when the nearest image changes.

    **Expected result.** The default results agree with chapter 02; differentiating strain independently gives the same stress.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why stress needs two derivatives
    For $E(R,C)$ and row-vector deformation $R'=RF$, $C'=CF$, the chain rule gives
    $$G_F=R^T\nabla_R E+C^T\nabla_C E,\qquad \sigma=\frac{G_F+G_F^T}{2V}.$$
    The second term accounts for periodic images. Differentiating the cell alone while holding Cartesian positions fixed is a different experiment. The formula assumes a nonsingular cell and a differentiable local image assignment.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    from metatomic import System

    mo.show_code(jax.config.update("jax_enable_x64", True), position="above")
    return System, jax, jnp, mo


@app.cell
def _(mo):
    mo.md(r"""
    ## Choose a separation
    In a standalone Python script, substitute `1.2` for `separation.value`.
    """)
    return


@app.cell
def _(mo):
    separation = mo.ui.slider(
        0.8, 1.6, step=0.02, value=1.2, label="Separation (angstrom)"
    )
    mo.show_code(separation, position="above")
    return (separation,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Build a JAX-backed System
    Integer types and Boolean PBC are metadata; positions and cell are the differentiable leaves.
    """)
    return


@app.cell
def _(System, jnp, mo, separation):
    def make_system(distance):
        return System(
            "angstrom",
            jnp.array([1, 1], dtype=jnp.int32),
            jnp.array([[1.0, 1.0, 1.0], [1.0 + distance, 1.0, 1.0]], dtype=jnp.float64),
            5.0 * jnp.eye(3),
            jnp.ones(3, dtype=jnp.bool_),
            arrays_backend="jax",
        )

    system = make_system(separation.value)
    mo.show_code(position="above")
    return (system,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Write a scalar energy and differentiate it
    `value_and_grad` returns a gradient tree shaped like the System. Negate position gradients for forces. For strain, combine position and cell terms before dividing by volume.
    """)
    return


@app.cell
def _(jax, jnp, mo):
    def energy_fn(system):
        d = system.positions[1] - system.positions[0]
        image = jax.lax.stop_gradient(jnp.round(d @ jnp.linalg.inv(system.cell)))
        d = d - image @ system.cell
        return 0.5 * 10.0 * (jnp.linalg.norm(d) - 1.0) ** 2

    @jax.jit
    def evaluate(system):
        energy, gradient = jax.value_and_grad(energy_fn)(system)
        forces = -gradient.positions
        # Under strain, both the row-vector positions and cell deform.
        dE_deformation = (
            system.positions.T @ gradient.positions + system.cell.T @ gradient.cell
        )
        stress = (dE_deformation + dE_deformation.T) / (
            2 * jnp.abs(jnp.linalg.det(system.cell))
        )
        return energy, forces, stress

    mo.show_code(position="above")
    return energy_fn, evaluate


@app.cell
def _(mo):
    mo.md(r"""
    ## Call the compiled evaluator
    Inspect the scalar and the two tensor shapes before connecting this function to an engine.
    """)
    return


@app.cell
def _(evaluate, mo, system):
    energy, forces, stress = evaluate(system)
    _display = mo.vstack(
        [
            mo.md(f"**E = {float(energy):.6f} eV**; forces in eV/Å, stress in eV/Å³"),
            forces,
            stress,
        ]
    )
    mo.show_code(_display, position="above")
    return energy, forces, stress


@app.cell
def _(mo):
    from metatomic_marimo.visualization import structure_panel

    mo.show_code(position="above")
    return (structure_panel,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualize energy and derivatives
    Move the separation slider and predict the direction of both forces before checking the arrows.
    """)
    return


@app.cell
def _(energy, forces, mo, stress, structure_panel, system):
    _display = structure_panel(system, energy=energy, forces=forces, stress=stress)
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Verify the strain convention independently
    Apply the same deformation to positions and cell, then differentiate with respect to that deformation. This provides a separate implementation of the stress calculation.
    """)
    return


@app.cell
def _(System, energy_fn, jax, jnp, mo, stress, system):
    def strained_energy(deformation, system):
        deformed = System(
            system.length_unit,
            system.types,
            system.positions @ deformation,
            system.cell @ deformation,
            system.pbc,
            arrays_backend="jax",
        )
        return energy_fn(deformed)

    strain_gradient = jax.grad(strained_energy)(jnp.eye(3), system)
    stress_from_strain = (strain_gradient + strain_gradient.T) / (
        2 * jnp.abs(jnp.linalg.det(system.cell))
    )
    assert jnp.allclose(stress, stress_from_strain, atol=1e-12)
    _display = mo.md(
        "✓ Explicit strain differentiation agrees with the position + cell formula."
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Boundary of current support:** no attached pair lists/custom data may enter JAX
    transformations. C pointer access is unavailable while tracing. For larger systems,
    provide fixed pair indices/image shifts separately and recompute vectors from the
    traced geometry. This notebook demonstrates JAX functions, not plugin registration
    or a shipped `JaxModel` class.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise
    Translate both atoms by the same vector. Energy, forces, and stress should remain unchanged. Next move a pair across the periodic boundary and retain the cell derivative.


    **Worked answer.** For a translation R → R + t, energy is invariant and the sum of atomic gradients is zero. Thus the translation term in Rᵀ∇R E vanishes. At a periodic image switch, the locally fixed-image derivative need not remain continuous.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce and continue
    Run `uv run marimo edit notebooks/03_jax_energy_forces_stress.py` from this project. Use the default controls to reproduce the reference result, then vary one input at a time. Replace `.value` controls with numeric values when copying to a script. Computation cells show their executed source; plotting helpers belong to this cookbook.

    This edition targets core System revision `b02b9ff3`, Python 3.12 and CPU float64. Source and dependencies are pinned in the guide; HTML exports preserve results but cannot rerun Python controls.

    [JAX automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html) · [Guide](/guide) · [API status](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
