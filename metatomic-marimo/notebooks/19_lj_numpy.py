import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 19 · Derive a NumPy Lennard–Jones kernel

    **Reference model · Executable tutorial**

    **Learning objective.** Derive the analytic pair derivative, implement cutoff masking, and verify a NumPy kernel against the native reference.

    **Prerequisites.** 02, 06 and 18; basic differentiation.

    Let $q=(\sigma/r)^6$. For $r<r_c$,
    $$U=4\epsilon(q^2-q)-U_{\rm LJ}(r_c),\qquad \nabla_{\mathbf d}U=24\epsilon(q-2q^2)\mathbf d/r^2.$$
    The derivative is with respect to $\mathbf d=\mathbf r_j-\mathbf r_i+\mathbf nC$. Its contributions to the position gradient are negative for $i$ and positive for $j$. Forces negate those gradients.


    **Expected result.** NumPy energies and gradients agree with the accepted C++ plugin to float64 tolerance, and the unshifted equilibrium separation remains $2^{1/6}\sigma$.

    ## Write the numerical kernel
    The implementation below is deliberately independent of the shared adapter. It rejects coincident pairs and makes the cutoff mask explicit. Unit and output-label handling belong to the model boundary.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from metatomic import Quantity
    from metatomic_marimo.lj_reference import NativeLJ, NumpyLJ, make_pair_system

    mo.show_code(position="above")
    return NativeLJ, NumpyLJ, Quantity, make_pair_system, mo, np, plt


@app.cell
def _(mo, np):
    def lj_numpy(vectors, sigma=1.0, epsilon=1.0, cutoff=3.0):
        squared = np.sum(vectors**2, axis=1)
        if np.any(squared <= 0):
            raise ValueError("pair distances must be positive")
        sixth = (sigma**2 / squared) ** 3
        cutoff_sixth = (sigma / cutoff) ** 6
        active = squared < cutoff**2
        energy = 4 * epsilon * (sixth**2 - sixth - cutoff_sixth**2 + cutoff_sixth)
        gradient = (24 * epsilon * (sixth - 2 * sixth**2) / squared)[:, None] * vectors
        return np.where(active, energy, 0.0), np.where(active[:, None], gradient, 0.0)

    vectors = np.array(
        [[1.2, 0.2, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.2, 0.0, 0.0]]
    )
    pair_energies, pair_gradients = lj_numpy(vectors)
    mo.show_code(pair_energies, position="above")
    return lj_numpy, pair_gradients, vectors


@app.cell
def _(mo):
    mo.md(r"""
    ## Check derivatives independently
    Finite differences should use a step small enough to approximate a derivative but large enough to avoid subtraction noise. The off-axis fixture below avoids both zero separation and the cutoff discontinuity. We compare all three vector components.
    """)
    return


@app.cell
def _(lj_numpy, mo, np, pair_gradients, vectors):
    step = 1e-6
    numerical_gradient = np.zeros(3)
    for axis in range(3):
        plus, minus = vectors[:1].copy(), vectors[:1].copy()
        plus[0, axis] += step
        minus[0, axis] -= step
        numerical_gradient[axis] = (lj_numpy(plus)[0][0] - lj_numpy(minus)[0][0]) / (
            2 * step
        )
    np.testing.assert_allclose(pair_gradients[0], numerical_gradient, atol=1e-8)
    mo.show_code(numerical_gradient, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Compare labeled outputs with the native model
    The local `NumpyLJ` adapter adds batch labels, atom selection and pair accounting to the same mathematics. It is a tutorial reference, not the pending public `NumpyModel`. The native call below tests the actual installed plugin, not a copied formula.
    """)
    return


@app.cell
def _(NativeLJ, NumpyLJ, Quantity, make_pair_system, mo, np):
    systems = [
        make_pair_system(1.2, transverse=0.2),
        make_pair_system(2.0),
        make_pair_system(3.2),
    ]
    requests = [
        Quantity(
            name="energy", unit="eV", sample_kind="system", gradients=["positions"]
        )
    ]
    python_outputs = NumpyLJ().execute(systems, None, requests)
    with NativeLJ() as native:
        cpp_outputs = native.execute(systems, None, requests)
    np.testing.assert_allclose(
        python_outputs[0].block().values, cpp_outputs[0].block().values, atol=1e-12
    )
    np.testing.assert_allclose(
        python_outputs[0].block().gradient("positions").values,
        cpp_outputs[0].block().gradient("positions").values,
        atol=1e-11,
    )
    mo.show_code(python_outputs[0].block().values, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Read the potential and its derivative together
    The force curve plotted here is the radial force on the second atom, $-dU/dr$. The zero near 1.122 Å is the potential minimum. The other zero region is beyond cutoff; it is not another stable well.
    """)
    return


@app.cell
def _(lj_numpy, mo, np, plt):
    radii = np.linspace(1.03, 3.2, 220)
    sweep_vectors = np.column_stack([radii, np.zeros_like(radii), np.zeros_like(radii)])
    sweep_energy, sweep_gradient = lj_numpy(sweep_vectors)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].plot(radii, sweep_energy)
    axes[1].plot(radii, -sweep_gradient[:, 0])
    for ax in axes:
        ax.axvline(3.0, color="0.5", linestyle=":", label="cutoff")
        ax.axhline(0.0, color="0.8", linewidth=0.7)
        ax.set_xlabel("Separation (Å)")
    axes[0].set_ylabel("Pair energy (eV)")
    axes[1].set_ylabel("Radial force on atom j (eV/Å)")
    fig.tight_layout()
    mo.show_code(fig, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    Why can the same half-list kernel be wrong by a factor of two when fed a full list?

    **Worked answer.** A full list contains both orientations of each physical pair. Sum with a factor of 1/2 or use only one orientation, and apply the same convention to derivatives. Do not halve energy while leaving forces unchanged.

    ## Reproduce and references
    Run `uv run marimo edit notebooks/19_lj_numpy.py`. The kernel, finite-difference probe and native comparison run on CPU float64. General `NumpyModel` registration, artifact serialization and engine integration are still future work.

    [Accepted LJ implementation](https://github.com/metatensor/metatomic/blob/12b24b14/metatomic-core/lj-plugin/lennard_jones.cpp) · [Label contract](/notebooks/outputs/) · [Guide](/guide)
    """)
    return


if __name__ == "__main__":
    app.run()
