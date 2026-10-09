import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 12 · GROMACS: a dimensional conversion test

    **Integration design note · Executable boundary fixture**

    **Learning objective.** Derive conversion factors for positions, forces and stress before changing a simulation module.

    **Prerequisites.** 02 and 07.

    **Scope.** C++ model API; this is a numerical boundary fixture, not a GROMACS run. The migration remains future work. The calculation below is complete and reproducible independently of the engine.

    ## Physical and data contract

    Choose model units eV and Å, and an engine boundary using kJ mol$^{-1}$ and nm. If $c_E$ converts energy and $c_L$ converts lengths, then forces scale by $c_E/c_L$ and stress by $c_E/c_L^3$. Applying only the energy factor to a force is a tenfold error here.

    The fixture concerns dimensions. GROMACS-specific virial signs, cell layout and distribution still require an adapter regression test.
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
    from ase import units

    energy_factor = 1.0 / (units.kJ / units.mol)
    length_factor = 0.1
    force_factor = energy_factor / length_factor
    stress_factor = energy_factor / length_factor**3
    converted = {
        "energy_kJ_mol": 0.2 * energy_factor,
        "distance_nm": 1.2 * length_factor,
        "force_kJ_mol_nm": 2.0 * force_factor,
        "stress_kJ_mol_nm3": 0.0192 * stress_factor,
    }
    np.testing.assert_allclose(
        converted["force_kJ_mol_nm"] * converted["distance_nm"],
        (2.0 * 1.2) * energy_factor,
    )
    mo.show_code(mo.ui.table([converted], selection=None), position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Migration specification

    | Stage | Required decision | Evidence before dynamics |
    |---|---|---|
    | Setup | Model/plugin lifetime, capabilities and type mapping | Clean load; unsupported type/device failures |
    | Geometry boundary | Units, cell layout and local/ghost ownership | Round-trip coordinates and periodic-image fixture |
    | Evaluation | Requested outputs and pair-list updates | Energy/force parity for multiple geometries |
    | Accumulation | Force combination, virial sign/order and units | Finite strain test; no double counting |
    | Restart/distribution | Persistent state and rank behavior | Restart and serial/parallel agreement |

    The first implementation milestone is fixed-geometry evaluation through the module. A successful trajectory is not a substitute for the boundary tests.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    What is the conversion factor for a force if positions are converted from Å to nm?

    **Worked answer.** It is 10 times the energy conversion factor. Stress requires 1000 times the energy factor because volume scales as the cube of length.

    ## Reproduce and references
    Run `uv run marimo edit notebooks/12_gromacs.py` in the pinned cookbook environment. The fixture uses CPU arrays; no engine build is required. For the full model contract and known native limitations, read [21](/notebooks/cpp-lj/).

    [GROMACS definitions and units](https://manual.gromacs.org/current/reference-manual/definitions.html) · [API roadmap](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
