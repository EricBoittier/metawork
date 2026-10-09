import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 02 · A NumPy teaching model: energy, forces, stress

    **Differentiation · Executable tutorial**

    **Learning objective.** Derive and evaluate energy, forces and configurational stress for a harmonic pair.

    **Prerequisites.** 00–01; vector calculus.

    We implement one harmonic bond, $E=\frac12 k(r-r_0)^2$, with $k=10$ eV/Å²
    and $r_0=1$ Å. This is not a realistic hydrogen potential.
    `HarmonicPair` below is an ordinary Python class, **not** the planned `NumpyModel`.
    It provides a working precursor to the custom-model tutorial.

    The minimum-image rule here is for an orthogonal periodic box. Avoid coincident
    atoms and image-boundary ties. This is one bond per cell, not a lattice sum.
    Stress is positive in tension, $\sigma=V^{-1}\partial E/\partial\epsilon$.

    **Expected result.** At r = 1.2 Å: E = 0.2 eV, forces ±2 eV/Å, and σxx = 0.0192 eV/Å³.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From a pair energy to forces and stress
    Let $\mathbf d=\mathbf r_1-\mathbf r_0-\mathbf nC$, with fixed image index $\mathbf n$, and $r=|\mathbf d|$. Then
    $$E=\tfrac12k(r-r_0)^2,\qquad \mathbf g=\frac{\partial E}{\partial\mathbf d}=k(r-r_0)\frac{\mathbf d}{r}.$$
    The atomic forces are $\mathbf F_0=\mathbf g$ and $\mathbf F_1=-\mathbf g$. With row-vector geometry deformed by $F=I+\epsilon$, the configurational stress is $\sigma=\mathbf d^T\mathbf g/V$ (outer product). It is positive in tension; hydrostatic pressure is $-\mathrm{tr}(\sigma)/3$. This excludes kinetic stress.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from metatomic import System

    mo.show_code(position="above")
    return System, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## Implement energy and analytic derivatives
    Differentiate the bond length using `d/r`. Atom 0 gets the positive radial derivative and atom 1 the negative. Stress uses the pair virial divided by cell volume. The class validates its restricted geometry.
    """)
    return


@app.cell
def _(mo, np):
    class HarmonicPair:
        def __init__(self, k=10.0, r0=1.0):
            self.k, self.r0 = k, r0

        def evaluate(self, system):
            positions, cell = np.asarray(system.positions), np.asarray(system.cell)
            if positions.shape != (2, 3) or not np.all(system.pbc):
                raise ValueError("This teaching model requires two atoms and full PBC")
            if not np.allclose(cell, np.diag(np.diag(cell))) or np.any(
                np.diag(cell) <= 0
            ):
                raise ValueError(
                    "This teaching model requires a positive orthogonal cell"
                )
            if system.length_unit != "angstrom":
                raise ValueError("This teaching model expects angstrom")
            d = positions[1] - positions[0]
            d = d - np.round(d @ np.linalg.inv(cell)) @ cell
            r = np.linalg.norm(d)
            if r == 0:
                raise ValueError("Coincident atoms are outside this tutorial's domain")
            energy = 0.5 * self.k * (r - self.r0) ** 2
            derivative = self.k * (r - self.r0) * d / r
            forces = np.stack([derivative, -derivative])
            stress = np.outer(d, derivative) / abs(np.linalg.det(cell))
            return float(energy), forces, stress

    numpy_model = HarmonicPair()
    mo.show_code(position="above")
    return (numpy_model,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Choose the geometry
    Replace `separation.value` with a number when copying this into a plain script.
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
    ## Evaluate your model
    The evaluator returns an energy scalar, an `(N, 3)` force array, and a `(3, 3)` stress tensor. Notebook 07 teaches the labeled output format.
    """)
    return


@app.cell
def _(System, mo, np, numpy_model, separation):
    def make_system(distance):
        return System(
            "angstrom",
            np.array([1, 1], dtype=np.int32),
            np.array([[1.0, 1.0, 1.0], [1.0 + distance, 1.0, 1.0]]),
            5.0 * np.eye(3),
            np.ones(3, dtype=np.bool_),
        )

    system = make_system(separation.value)
    energy, forces, stress = numpy_model.evaluate(system)
    _display = mo.vstack(
        [
            mo.md(f"**E = {energy:.6f} eV**; forces in eV/Å, stress in eV/Å³"),
            forces,
            stress,
        ]
    )
    mo.show_code(_display, position="above")
    return energy, forces, make_system, stress, system


@app.cell
def _(mo):
    from metatomic_marimo.visualization import structure_panel

    mo.show_code(position="above")
    return (structure_panel,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualize the prediction
    The arrow scale affects the drawing only. Expand the tables to inspect numerical values.
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
    ## Sweep a parameter
    Construct several independent Systems and evaluate their energies. The minimum should occur at the model parameter `r0`.
    """)
    return


@app.cell
def _(energy, make_system, mo, np, numpy_model, plt, separation):
    radii = np.linspace(0.8, 1.6, 81)
    curve = np.array([numpy_model.evaluate(make_system(r))[0] for r in radii])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(radii, curve)
    ax.scatter([separation.value], [energy], color="tab:red")
    ax.set(
        xlabel="Separation (angstrom)",
        ylabel="Energy (eV)",
        title="Harmonic pair energy",
    )
    fig.tight_layout()
    mo.show_code(fig, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    For the default separation, expect 0.2 eV, forces `[+2, 0, 0]` and `[-2, 0, 0]`
    eV/Å, and `stress[0, 0] = 0.0192` eV/Å³. Pressure is `-trace(stress)/3`.
    The model returns a tutorial tuple; standard metatomic model outputs will require
    TensorMaps and the eventual model base-class contract.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise
    Set k=20 while keeping r0=1 and the geometry fixed. Predict how energy, force, and stress scale. At r=r0, all three should vanish.


    **Worked answer.** All three predictions are linear in k at fixed geometry. Doubling k gives E = 0.4 eV, force magnitudes 4 eV/Å and σxx = 0.0384 eV/Å³.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce and continue
    Run `uv run marimo edit notebooks/02_numpy_model.py` from this project. Use the default controls to reproduce the reference result, then vary one input at a time. Replace `.value` controls with numeric values when copying to a script. Computation cells show their executed source; plotting helpers belong to this cookbook.

    This edition targets core System revision `b02b9ff3`, Python 3.12 and CPU float64. Source and dependencies are pinned in the guide; HTML exports preserve results but cannot rerun Python controls.

    [Published model tutorial](https://docs.metatensor.org/metatomic/latest/examples/1-export-atomistic-model.html) · [Guide](/guide) · [API status](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
