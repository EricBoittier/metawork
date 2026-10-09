import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 11 · LAMMPS: forces, virial and ownership

    **Integration design note · Executable boundary fixture**

    **Learning objective.** Translate a configurational stress into an extensive virial, then assign responsibilities to pair, fix and compute interfaces.

    **Prerequisites.** 02, 07 and 18.

    **Scope.** C++ API; MPI and the LAMMPS plugin are not executed here. The migration remains future work. The calculation below is complete and reproducible independently of the engine.

    ## Physical and data contract

    For this note define $W=\sum_i \mathbf r_i\otimes\mathbf F_i=-V\sigma$ for an isolated pair represented without image ambiguity. The configurational pressure tensor is $W/V$; total pressure also contains a kinetic contribution. Do not infer an engine’s tensor order or conversion constants from the word “virial”.

    This fixture uses the chapter-02 harmonic pair so its strain response is known. It does not request unsupported stress from the native LJ model.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np

    mo.show_code(position="above")
    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    ## Run the boundary fixture
    Read the dimensional or indexing argument first, then run the assertion. Passing this calculation validates only the stated boundary property.
    """)
    return


@app.cell
def _(mo, np):
    positions = np.array([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]])
    forces = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    volume = 125.0
    stress = np.diag([0.0192, 0.0, 0.0])
    virial = positions.T @ forces
    np.testing.assert_allclose(virial, -volume * stress, atol=1e-12)
    np.testing.assert_allclose(
        (positions + [7.0, -2.0, 3.0]).T @ forces, virial, atol=1e-12
    )
    pressure_config = np.trace(virial) / (3 * volume)
    _display = mo.md(
        f"Wxx = {virial[0, 0]:.4f} eV; configurational pressure = {pressure_config:.4f} eV/Å³."
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Migration specification

    | Interface | Responsibility | Completion evidence |
    |---|---|---|
    | `pair_style metatomic` | Load/retain model; type and local/ghost mapping; pair requests; accumulate owned energy/forces | Serial LJ parity; no duplicate pair counting; one-rank/multi-rank agreement |
    | `fix metatomic` | Preserve documented invocation points and model state across repeated steps | Restart and scheduling tests; additive force contributions checked |
    | `compute metatomic` | Map requested global/per-atom quantities and refresh cached results | Correct output labels, ownership and update frequency |
    | Documentation | Build, plugin options, units, input examples and migration behavior | Commands checked against the implemented version |

    **Decision needed before implementation:** partitioning and ghost-force accumulation must be explicit. The fixture above cannot establish parallel correctness. Start with the [native LJ reference](/notebooks/lj-c/); add serial pair evaluation before fix/compute and MPI.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    Translate the coordinates but keep the forces fixed. Why is W unchanged?

    **Worked answer.** The extra term is t ⊗ ΣF. It vanishes for this force-balanced system. A nonzero total force makes this simple origin-based expression depend on the origin. Periodic implementations must use image-consistent pair vectors.

    ## Reproduce and references
    Run `uv run marimo edit notebooks/11_lammps.py` in the pinned cookbook environment. The fixture uses CPU arrays; no engine build is required. For the full model contract and known native limitations, read [21](/notebooks/cpp-lj/).

    [LAMMPS pressure documentation](https://docs.lammps.org/compute_pressure.html) · [API roadmap](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
