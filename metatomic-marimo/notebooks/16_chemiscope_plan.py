import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 16 · Chemiscope: a faithful prediction dataset

    **Integration design note · Executable boundary fixture**

    **Learning objective.** Map quantities to structures or atoms without inventing a decomposition, and verify that display conversion cannot mutate the model inputs.

    **Prerequisites.** 00–02 and 07.

    **Scope.** Core System works; a public upstream converter remains future work. The migration remains future work. The calculation below is complete and reproducible independently of the engine.

    ## Physical and data contract

    A dataset needs a clear target for each property. Total energy, volume and stress belong to a structure; each force vector belongs to an atom. Atom-resolved energy is meaningful only when the model actually provides that partition.

    The local converter below copies geometry and predictions. Its red arrows use a display scale in Å per (eV/Å); changing that scale must not change numerical force values.
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
    from metatomic_marimo.lj_reference import make_pair_system
    from metatomic_marimo.visualization import make_dataset, structure_panel

    system = make_pair_system(1.2)
    energy, forces = 0.2, np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    stress = np.diag([2.4 / abs(np.linalg.det(system.cell)), 0.0, 0.0])
    before = system.positions.copy()
    dataset = make_dataset(
        [system], energies=[energy], forces=[forces], stresses=[stress]
    )
    assert dataset["properties"]["energy"]["target"] == "structure"
    assert len(dataset["structures"]) == 1
    np.testing.assert_array_equal(system.positions, before)
    _display = structure_panel(system, energy=energy, forces=forces, stress=stress)
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Migration specification

    | Data | Mapping rule | Regression check |
    |---|---|---|
    | Geometry/cell | Preserve units, PBC and atom ordering | Coordinate and cell equality |
    | Energy/stress | Structure properties | One value per frame; correct tensor components |
    | Forces | Atom vectors and explicit display scale | Direction/magnitude checks |
    | Trajectory | Align every property with its frame | Different frame counts fail early |
    | Backend arrays | Copy outside differentiation | Inputs unchanged after display |

    The displayed numbers are the harmonic fixture from chapter 02; the two-atom builder supplies geometry only. This is a working local display adapter, not a completed upstream migration. A public API should specify accepted labels, units, optional fields and array backends.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    Would total_energy / atom_count be a safe way to display model atomic energies?

    **Worked answer.** It is a uniform visualization convention, not the model’s atomic energy partition. Keep the total as a structure property unless a real atom-resolved quantity was requested and returned.

    ## Reproduce and references
    Run `uv run marimo edit notebooks/16_chemiscope_plan.py` in the pinned cookbook environment. The fixture uses CPU arrays; no engine build is required. For the full model contract and known native limitations, read [21](/notebooks/cpp-lj/).

    [Chemiscope Python reference](https://chemiscope.org/docs/python/reference.html) · [API roadmap](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
