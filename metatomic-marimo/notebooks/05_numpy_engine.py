import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 05 · A small NumPy engine: relax a two-atom structure

    **Simulation · Executable tutorial**

    **Learning objective.** Implement force-driven relaxation and distinguish convergence from reaching the iteration limit.

    **Prerequisites.** 02; loops and arrays.

    An engine owns geometry and repeatedly asks a model for energy and forces.
    Here we implement fixed-cell steepest descent, not molecular dynamics: the step
    size has units Å²/eV, there are no masses or timesteps, and no barostat.

    The callable `bond_energy_forces` is a teaching interface, not the pending
    metatomic engine/model protocol. All geometry updates happen on owned NumPy
    arrays, and each evaluation constructs a new core `System`.

    **Expected result.** With step size 0.02 Å²/eV, the separation error contracts by 0.6 per update until the force tolerance is met.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step size and stability
    Write $\delta r=r-r_0$. Updating both endpoints with $R\leftarrow R+\alpha F$ gives
    $$\delta r_{n+1}=(1-2\alpha k)\delta r_n.$$
    Thus $0<\alpha k<1$ gives contraction in this local harmonic problem; beyond $\alpha k=1/2$ the separation error changes sign. This analysis motivates the slider range and explains why a force-driven update is not automatically convergent for arbitrary models.
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
    ## Write a force-driven relaxation loop
    Update positions with `positions += step_size * forces`, then construct a fresh System. The loop tests force convergence and records evaluated frames, including the final state.
    """)
    return


@app.cell
def _(System, mo, np):
    def bond_energy_forces(system):
        d = system.positions[1] - system.positions[0]
        d = d - np.round(d @ np.linalg.inv(system.cell)) @ system.cell
        r = np.linalg.norm(d)
        derivative = 10.0 * (r - 1.0) * d / r
        return 0.5 * 10.0 * (r - 1.0) ** 2, np.stack([derivative, -derivative])

    def relax(initial, model, step_size=0.02, max_steps=100, tolerance=1e-6):
        positions = np.array(initial.positions, copy=True)
        cell = np.array(initial.cell, copy=True)
        types = np.array(initial.types, copy=True)
        pbc = np.array(initial.pbc, copy=True)
        history = []
        converged = False
        for step in range(max_steps + 1):
            current = System(
                initial.length_unit,
                types.copy(),
                positions.copy(),
                cell.copy(),
                pbc.copy(),
            )
            energy, forces = model(current)
            max_force = float(np.max(np.linalg.norm(forces, axis=1)))
            history.append(
                {
                    "step": step,
                    "system": current,
                    "forces": np.array(forces, copy=True),
                    "energy_eV": float(energy),
                    "max_force_eV_per_A": max_force,
                }
            )
            if max_force < tolerance:
                converged = True
                break
            if step < max_steps:
                positions += step_size * forces
        return current, history, converged

    mo.show_code(position="above")
    return bond_energy_forces, relax


@app.cell
def _(mo):
    mo.md(r"""
    ## Choose a relaxation step
    This parameter has units angstrom²/eV. It is not an MD timestep.
    """)
    return


@app.cell
def _(mo):
    step_size = mo.ui.slider(
        0.005, 0.08, step=0.005, value=0.02, label="Relaxation step (angstrom²/eV)"
    )
    mo.show_code(step_size, position="above")
    return (step_size,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Run relaxation and report convergence
    Always distinguish a converged structure from one that merely reached a step limit.
    """)
    return


@app.cell
def _(System, bond_energy_forces, mo, np, relax, step_size):
    initial = System(
        "angstrom",
        np.array([1, 1], dtype=np.int32),
        np.array([[1.0, 1.0, 1.0], [2.4, 1.0, 1.0]]),
        5.0 * np.eye(3),
        np.ones(3, dtype=np.bool_),
    )
    relaxed, history, converged = relax(
        initial, bond_energy_forces, step_size=step_size.value
    )
    _display = mo.md(
        f"**Converged: {converged}** after {len(history) - 1} updates; final energy {history[-1]['energy_eV']:.3e} eV."
    )
    mo.show_code(_display, position="above")
    return (history,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Plot convergence
    Track both energy and maximum atomic force. A decreasing energy alone is not a force convergence criterion.
    """)
    return


@app.cell
def _(history, mo, plt):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].plot(
        [row["step"] for row in history], [row["energy_eV"] for row in history]
    )
    axes[0].set(xlabel="Step", ylabel="Energy (eV)")
    axes[1].semilogy(
        [row["step"] for row in history],
        [max(row["max_force_eV_per_A"], 1e-16) for row in history],
    )
    axes[1].set(xlabel="Step", ylabel="Maximum force (eV/angstrom)")
    fig.tight_layout()
    mo.show_code(fig, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    To extend this into a production engine: validate capabilities and units, build
    requested neighbor lists, request standard output TensorMaps, handle devices,
    check convergence, and rebuild pairs as atoms/cell move. The analogous Torch/JAX
    engines can retain geometry as backend tensors, with kernels from notebooks 03/04;
    this notebook implements only the NumPy loop. See `ROADMAP.md` for upstream blockers.
    """)
    return


@app.cell
def _(mo):
    from metatomic_marimo.visualization import structure_panel, trajectory_panel

    mo.show_code(position="above")
    return structure_panel, trajectory_panel


@app.cell
def _(mo):
    mo.md(r"""
    ## Inspect intermediate frames
    Select a step in the map to connect the force direction with the atomic update.
    """)
    return


@app.cell
def _(history, mo, trajectory_panel):
    _display = trajectory_panel(
        [row["system"] for row in history],
        [row["energy_eV"] for row in history],
        [row["forces"] for row in history],
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Inspect the final structure
    The bond should approach 1 angstrom. Notebook 08 moves from optimization to physical dynamics and adds JAX/Torch backends.
    """)
    return


@app.cell
def _(history, mo, structure_panel):
    _final = history[-1]
    _display = structure_panel(
        _final["system"], energy=_final["energy_eV"], forces=_final["forces"]
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise
    Reduce max_steps to 1 and confirm converged=False. Increase the step size and inspect oscillation or divergence rather than assuming every force-based update improves the geometry.


    **Worked answer.** For two free endpoints, δrₙ₊₁ = (1 − 2αk)δrₙ. Stability requires 0 < αk < 1; monotone separation needs αk ≤ 1/2. At α = 0.02 and k = 10, one update cannot satisfy a 10⁻⁶ force tolerance from r = 1.4 Å.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce and continue
    Run `uv run marimo edit notebooks/05_numpy_engine.py` from this project. Use the default controls to reproduce the reference result, then vary one input at a time. Replace `.value` controls with numeric values when copying to a script. Computation cells show their executed source; plotting helpers belong to this cookbook.

    This edition targets core System revision `b02b9ff3`, Python 3.12 and CPU float64. Source and dependencies are pinned in the guide; HTML exports preserve results but cannot rerun Python controls.

    [ASE optimization](https://docs.ase-lib.org/ase/optimize.html) · [Guide](/guide) · [API status](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
