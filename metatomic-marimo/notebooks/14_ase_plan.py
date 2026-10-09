import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 14 · ASE: property requests and cache invalidation

    **Integration design note · Executable boundary fixture**

    **Learning objective.** Test how stored predictions relate to an atomic configuration before designing the new calculator boundary.

    **Prerequisites.** 07–08.

    **Scope.** Public Python execution API; the production metatomic calculator is not rewritten here. The migration remains future work. The calculation below is complete and reproducible independently of the engine.

    ## Physical and data contract

    An energy belongs to a particular geometry and model state. A calculator must invalidate stored predictions when that state changes. ASE’s single-point calculator is useful for testing this boundary: it stores known results, and refuses to evaluate a modified configuration.

    The example below is a cache test, not a force evaluator. The executable three-backend teaching calculator is in [08](/notebooks/dynamics/).
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
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.calculators.calculator import PropertyNotImplementedError

    atoms = Atoms(
        "H2", positions=[[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]], cell=5 * np.eye(3), pbc=True
    )
    stored_forces = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=0.2,
        forces=stored_forces,
        stress=np.array([0.0192, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    initial_energy = atoms.get_potential_energy()
    atoms.positions[1, 0] += 0.1
    try:
        atoms.get_potential_energy()
    except PropertyNotImplementedError:
        cache_invalidated = True
    else:
        cache_invalidated = False
    assert cache_invalidated
    _display = mo.md(
        f"Stored energy: {initial_energy:.1f} eV. Changed geometry rejected: **{cache_invalidated}**."
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Migration specification

    | ASE request | Core data required | Adapter obligation |
    |---|---|---|
    | energy | System energy | Convert units; return scalar |
    | forces | Position gradient | Negate gradient; preserve atom ordering |
    | stress | Strain gradient and nonzero cell volume | Divide by volume; convert to ASE ordering |
    | Repeated call | Same geometry and relevant model state | Reuse only valid cached results |

    The production migration requires a real loaded model and explicit capabilities. Reject unavailable stress rather than returning zeros. Test positions, cell, PBC, species, model parameters and requested properties as cache inputs.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    Why not reuse the energy after moving one atom if the calculator still holds it?

    **Worked answer.** The value was computed for the previous configuration. A cache is valid only for its recorded inputs. A production calculator must recompute; this single-point fixture correctly refuses because it has no model.

    ## Reproduce and references
    Run `uv run marimo edit notebooks/14_ase_plan.py` in the pinned cookbook environment. The fixture uses CPU arrays; no engine build is required. For the full model contract and known native limitations, read [21](/notebooks/cpp-lj/).

    [ASE calculators](https://docs.ase-lib.org/ase/calculators/calculators.html) · [API roadmap](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
