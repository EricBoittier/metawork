import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 13 · PLUMED: the bias-force chain rule

    **Integration design note · Executable boundary fixture**

    **Learning objective.** Check a scalar observable’s coordinate derivative before connecting it to a biasing Action.

    **Prerequisites.** 02–03; chain rule.

    **Scope.** C++ API; PLUMED itself is not loaded in this notebook. The migration remains future work. The calculation below is complete and reproducible independently of the engine.

    ## Physical and data contract

    For a scalar observable $s(R)$ and harmonic bias $B=\frac12\kappa(s-s_0)^2$,
    $$F_i^{\mathrm{bias}}=-\kappa(s-s_0)\frac{\partial s}{\partial R_i}.$$
    Here $s$ is the distance between two atoms. This minimal example isolates the chain rule: a model output used as an observable must supply its derivative, not a force with an assumed sign.
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
    r = np.array([[0.0, 0.0, 0.0], [1.2, 0.2, 0.0]])
    kappa, target = 5.0, 1.0

    def bias_energy(r):
        return 0.5 * kappa * (np.linalg.norm(r[1] - r[0]) - target) ** 2

    vector = r[1] - r[0]
    cv = np.linalg.norm(vector)
    cv_gradient = np.stack([-vector / cv, vector / cv])
    bias_forces = -kappa * (cv - target) * cv_gradient
    step = 1e-6
    numeric = np.zeros_like(r)
    for _atom in range(2):
        for _axis in range(3):
            _plus, _minus = r.copy(), r.copy()
            _plus[_atom, _axis] += step
            _minus[_atom, _axis] -= step
            numeric[_atom, _axis] = -(bias_energy(_plus) - bias_energy(_minus)) / (
                2 * step
            )
    np.testing.assert_allclose(bias_forces, numeric, atol=1e-8)
    mo.show_code(bias_forces, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Migration specification

    | Boundary | Migration requirement | Test |
    |---|---|---|
    | Action setup | Load model and define supported observables/units | Existing Action input compatibility |
    | Atom selection | Preserve environment and selection labels | Permutation and partial selection |
    | Atomic derivatives | Apply the bias chain rule once | Finite differences of biased energy |
    | Box derivatives | Convert the declared strain derivative with the correct convention | Homogeneous deformation test |
    | Evaluation cache | Invalidate on geometry, inputs and model-state changes | Two consecutive distinct configurations |

    Begin with one scalar observable. The native LJ reference does not advertise strain gradients, so it cannot currently validate a box-derivative path by itself.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    What happens if the model supplies forces but the Action interprets them as ds/dR?

    **Worked answer.** The bias-force sign is reversed. For an energy observable, its coordinate derivative is the negative of its physical force; the adapter must represent that distinction explicitly.

    ## Reproduce and references
    Run `uv run marimo edit notebooks/13_plumed.py` in the pinned cookbook environment. The fixture uses CPU arrays; no engine build is required. For the full model contract and known native limitations, read [21](/notebooks/cpp-lj/).

    [PLUMED harmonic restraint](https://www.plumed.org/doc-v2.10/user-doc/html/_r_e_s_t_r_a_i_n_t.html) · [API roadmap](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
