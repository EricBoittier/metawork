import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 15 · i-PI: energy, force and virial units

    **Integration design note · Executable boundary fixture**

    **Learning objective.** Build a dimensionally consistent request/response fixture before testing a driver connection.

    **Prerequisites.** 02 and 07.

    **Scope.** Public Python execution API; no socket connection is opened. The migration remains future work. The calculation below is complete and reproducible independently of the engine.

    ## Physical and data contract

    The i-PI boundary sends coordinates/cell and receives potential energy, forces and a potential virial. Its internal units are atomic units. Starting from eV and Å, convert energy with the Hartree and lengths with the Bohr radius; the force unit is Hartree/Bohr.

    This fixture defines the potential virial as $W=-V\sigma$ and checks its dimensions. A real driver must additionally verify cell serialization, tensor ordering and the protocol convention. It must not send a stress density where an extensive virial is expected.
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
    from ase.units import Bohr, Hartree

    energy_eV = 0.2
    force_eV_A = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    stress_eV_A3 = np.diag([0.0192, 0.0, 0.0])
    volume_A3 = 125.0
    response_atomic_units = {
        "energy": energy_eV / Hartree,
        "forces": force_eV_A * Bohr / Hartree,
        "virial": -volume_A3 * stress_eV_A3 / Hartree,
    }
    np.testing.assert_allclose(
        response_atomic_units["forces"] * Hartree / Bohr, force_eV_A
    )
    np.testing.assert_allclose(
        response_atomic_units["virial"] * Hartree, -volume_A3 * stress_eV_A3
    )
    mo.show_code(response_atomic_units, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Migration specification

    | Stage | Proposed responsibility | Completion test |
    |---|---|---|
    | Initialization | Retain model and negotiate inputs | Missing dependency/type errors |
    | Receive | Decode geometry and units; construct System | Recorded coordinate/cell fixture |
    | Evaluate | Rebuild requested pairs and invoke the public executor | Independent energy/force/virial values |
    | Reply | Encode values in protocol units/order | Deterministic request/response comparison |
    | Repeated requests | Own buffers and handle failures/reconnects | Two geometries, disconnect and shutdown |

    One validated request is the first milestone. Multi-client dynamics and path-integral workloads come after protocol correctness, not before.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    Why is there no factor of Bohr cubed in the virial conversion above?

    **Worked answer.** Multiplying stress by the volume cancels length dimensions. The virial has energy units. A stress value alone would require a volume-unit conversion.

    ## Reproduce and references
    Run `uv run marimo edit notebooks/15_ipi.py` in the pinned cookbook environment. The fixture uses CPU arrays; no engine build is required. For the full model contract and known native limitations, read [21](/notebooks/cpp-lj/).

    [i-PI units and conventions](https://docs.ipi-code.org/units.html) · [API roadmap](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
