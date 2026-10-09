import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 08 · An ASE engine adapter for NumPy, JAX, and Torch

    **Simulation · Executable tutorial**

    **Learning objective.** Connect three evaluators to an ASE calculator and diagnose integration error using total energy.

    **Prerequisites.** 02–05; Newtonian dynamics.


    The [published ASE tutorial](https://docs.metatensor.org/metatomic/latest/examples/2-running-ase-md.html)
    connects an exported TorchScript model to ASE. This original example connects
    our teaching kernels directly while the new Python loader/base classes are pending.
    `TutorialCalculator` below is local tutorial code, **not** `MetatomicCalculator`.
    We use NVE dynamics with zero initial velocities and a stretched bond; potential
    energy becomes kinetic energy. This is not a thermostat or a material simulation.

    **Expected result.** The 60-step, 0.05 fs trajectory has total-energy deviation below 10⁻⁴ eV; potential and kinetic energies exchange.

    ## Supply energy, forces, and stress using your preferred backend
    All branches use the same one-bond potential in eV and angstrom. JAX/Torch
    compute derivatives; NumPy uses the analytic result. Geometry stays outside
    DLPack once an autograd graph starts.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import jax
    import jax.numpy as jnp
    import torch
    import matplotlib.pyplot as plt
    from ase import Atoms, units
    from ase.calculators.calculator import Calculator, all_changes
    from ase.md.verlet import VelocityVerlet
    from ase.stress import full_3x3_to_voigt_6_stress
    from metatomic import System
    from metatomic_marimo.visualization import trajectory_panel

    mo.show_code(jax.config.update("jax_enable_x64", True), position="above")
    return (
        Atoms,
        Calculator,
        System,
        VelocityVerlet,
        all_changes,
        full_3x3_to_voigt_6_stress,
        jax,
        jnp,
        mo,
        np,
        plt,
        torch,
        trajectory_panel,
        units,
    )


@app.cell
def _(jax, jnp, mo, np, torch):
    def array_energy(positions, cell, xp):
        d = positions[1] - positions[0]
        d = d - xp.round(d @ xp.linalg.inv(cell)) @ cell
        return 5.0 * (xp.linalg.norm(d) - 1.0) ** 2

    jax_derivatives = jax.jit(
        jax.value_and_grad(lambda r, c: array_energy(r, c, jnp), argnums=(0, 1))
    )

    def evaluate_system(system, backend):
        r, c = np.array(system.positions), np.array(system.cell)
        if backend == "numpy":
            d = r[1] - r[0]
            d = d - np.round(d @ np.linalg.inv(c)) @ c
            distance = np.linalg.norm(d)
            derivative = 10.0 * (distance - 1.0) * d / distance
            return (
                float(5.0 * (distance - 1.0) ** 2),
                np.stack([derivative, -derivative]),
                np.outer(d, derivative) / abs(np.linalg.det(c)),
            )
        if backend == "jax":
            e, (gr, gc) = jax_derivatives(jnp.array(r), jnp.array(c))
            e, gr, gc = float(e), np.asarray(gr), np.asarray(gc)
        elif backend == "torch":
            tr = torch.tensor(r, requires_grad=True)
            tc = torch.tensor(c, requires_grad=True)
            te = array_energy(tr, tc, torch)
            tgr, tgc = torch.autograd.grad(te, (tr, tc))
            e, gr, gc = float(te.detach()), tgr.detach().numpy(), tgc.detach().numpy()
        else:
            raise ValueError(f"Unknown backend: {backend}")
        strain = r.T @ gr + c.T @ gc
        return e, -gr, (strain + strain.T) / (2 * abs(np.linalg.det(c)))

    mo.show_code(position="above")
    return (evaluate_system,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Implement the engine boundary
    ASE calls `calculate` when the atomic state or requested properties change.
    The adapter creates a fresh System, validates the teaching model's domain, calls
    the evaluator, and places results in ASE's expected dictionary. ASE's six stress
    components are ordered xx, yy, zz, yz, xz, xy; use its conversion helper.
    """)
    return


@app.cell
def _(
    Calculator,
    System,
    all_changes,
    evaluate_system,
    full_3x3_to_voigt_6_stress,
    mo,
    np,
):
    class TutorialCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def __init__(self, backend="numpy", **kwargs):
            super().__init__(**kwargs)
            self.backend = backend

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            a = self.atoms
            if len(a) != 2 or not np.all(a.numbers == 1) or not np.all(a.pbc):
                raise ValueError(
                    "This teaching calculator requires two H atoms and full PBC"
                )
            if not np.allclose(a.cell.array, np.diag(np.diag(a.cell.array))) or np.any(
                np.diag(a.cell.array) <= 0
            ):
                raise ValueError("Use a positive orthogonal cell")
            if a.get_distance(0, 1, mic=True) < 1e-12:
                raise ValueError("Coincident atoms are unsupported")
            system = System(
                "angstrom",
                a.numbers.astype(np.int32),
                a.positions.copy(),
                a.cell.array.copy(),
                a.pbc.copy(),
            )
            energy, forces, stress = evaluate_system(system, self.backend)
            self.results = {
                "energy": energy,
                "forces": forces,
                "stress": full_3x3_to_voigt_6_stress(stress),
            }

    mo.show_code(position="above")
    return (TutorialCalculator,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Select a backend and integrate the equations of motion
    Unlike the relaxation step in notebook 05, this timestep has physical time units.
    Velocities, masses, and kinetic energy now matter. Use a sufficiently small timestep
    and monitor **total** energy, not just potential energy, to assess NVE integration.
    """)
    return


@app.cell
def _(mo):
    backend_choice = mo.ui.dropdown(
        ["numpy", "jax", "torch"], value="numpy", label="Energy/force backend"
    )
    mo.show_code(backend_choice, position="above")
    return (backend_choice,)


@app.cell
def _(
    Atoms,
    System,
    TutorialCalculator,
    VelocityVerlet,
    backend_choice,
    mo,
    np,
    units,
):
    def run_dynamics(backend, steps=60, timestep_fs=0.05):
        atoms = Atoms(
            "H2",
            positions=[[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]],
            cell=5 * np.eye(3),
            pbc=True,
        )
        atoms.set_velocities(np.zeros((2, 3)))
        atoms.calc = TutorialCalculator(backend=backend)
        dynamics = VelocityVerlet(atoms, timestep_fs * units.fs)
        records = []
        for step in range(steps + 1):
            e = atoms.get_potential_energy()
            forces = atoms.get_forces()
            snapshot = System(
                "angstrom",
                atoms.numbers.astype(np.int32),
                atoms.positions.copy(),
                atoms.cell.array.copy(),
                atoms.pbc.copy(),
            )
            records.append(
                {
                    "system": snapshot,
                    "energy": e,
                    "forces": forces.copy(),
                    "time_fs": step * timestep_fs,
                    "kinetic": atoms.get_kinetic_energy(),
                    "stress": atoms.get_stress(voigt=False),
                }
            )
            if step < steps:
                dynamics.run(1)
        return records

    records = run_dynamics(backend_choice.value)
    total_energy = np.array([row["energy"] + row["kinetic"] for row in records])
    energy_drift = float(np.max(np.abs(total_energy - total_energy[0])))
    _display = mo.md(
        f"Backend: **{backend_choice.value}**. Maximum total-energy deviation over 3 fs: **{energy_drift:.3g} eV**."
    )
    mo.show_code(_display, position="above")
    return records, total_energy


@app.cell
def _(mo):
    mo.md(r"""
    ## Inspect the trajectory and energy exchange
    The structure map uses frame index on x, potential energy on y, and maximum force
    as color. The static chart separates potential/kinetic/total energy. Increase the
    timestep to see integration error grow; changing backend should leave the physics unchanged.
    """)
    return


@app.cell
def _(mo, records, trajectory_panel):
    _display = trajectory_panel(
        [r["system"] for r in records],
        [r["energy"] for r in records],
        [r["forces"] for r in records],
        stresses=[r["stress"] for r in records],
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo, plt, records, total_energy):
    fig, ax = plt.subplots(figsize=(7, 3))
    times = [r["time_fs"] for r in records]
    ax.plot(times, [r["energy"] for r in records], label="potential")
    ax.plot(times, [r["kinetic"] for r in records], label="kinetic")
    ax.plot(times, total_energy, label="total")
    ax.set(xlabel="Time (fs)", ylabel="Energy (eV)")
    ax.legend()
    fig.tight_layout()
    mo.show_code(fig, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What changes for a production model?
    Replace `evaluate_system` with the eventual public model call. Honor capabilities,
    requested pairs, output labels, device placement, dtype, and unit conversion.
    This adapter intentionally supports a tiny known system; it is not a substitute
    for the existing ASE integration or a generic model loader.

    **Try it:** halve the timestep at the same total simulated time. Verify improved
    energy conservation, then switch all three backends and compare trajectories.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    How would you test the timestep error without changing the physical duration of the trajectory?

    **Worked answer.** Compare dt and dt/2 at the same physical duration (double the number of steps). Velocity Verlet has second-order global accuracy for this smooth problem; look for the expected reduction in energy-error amplitude, not monotonically decreasing total energy.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce and continue
    Run `uv run marimo edit notebooks/08_ase_dynamics.py` from this project. Use the default controls to reproduce the reference result, then vary one input at a time. Replace `.value` controls with numeric values when copying to a script. Computation cells show their executed source; plotting helpers belong to this cookbook.

    This edition targets core System revision `b02b9ff3`, Python 3.12 and CPU float64. Source and dependencies are pinned in the guide; HTML exports preserve results but cannot rerun Python controls.

    [Published ASE tutorial](https://docs.metatensor.org/metatomic/latest/examples/2-running-ase-md.html) · [Guide](/guide) · [API status](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
