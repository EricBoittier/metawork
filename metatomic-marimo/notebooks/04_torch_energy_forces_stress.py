import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 04 · Torch: fresh autograd leaves and a compiled tensor kernel

    **Differentiation · Executable tutorial**

    **Learning objective.** Place the autograd boundary correctly and verify energy, forces and stress before and after graph capture.

    **Prerequisites.** 02; basic Torch tensors and nn.Module.

    Use the **core** `metatomic.System` with Torch arrays. DLPack storage sharing does
    not preserve a model's autograd history. Extract tensors once, clone them, and
    start differentiation on these local leaves. Do not construct a new core `System`
    from intermediate autograd tensors.

    `HarmonicPair` is a normal `torch.nn.Module`, not the proposed `TorchModel`.
    The scope is one harmonic bond in a fully periodic orthogonal cell, away from
    image switches or coincident atoms; it is not a trained hydrogen potential.

    **Expected result.** Eager and aot_eager predictions agree within float64 tolerances for the default geometry.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The differentiation boundary
    Differentiate the energy with respect to the same retained tensors used to evaluate it:
    $$\mathbf F=-\nabla_R E,\qquad \sigma=\mathrm{sym}(R^T\nabla_R E+C^T\nabla_C E)/V.$$
    The model-to-engine boundary may transport storage without retaining graph history. This example starts new leaves deliberately. The final conversion for visualization happens after both derivatives are obtained.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch
    from metatomic import System

    mo.show_code(position="above")
    return System, mo, torch


@app.cell
def _(mo):
    mo.md(r"""
    ## Define a module and its evaluation boundary
    Keep the compiled kernel tensor-only. Extract geometry once, clone it into fresh autograd leaves, and request both position and cell gradients. `aot_eager` checks capture and autograd, not optimized speed.
    """)
    return


@app.cell
def _(mo, torch):
    class HarmonicPair(torch.nn.Module):
        def forward(self, positions, cell):
            d = positions[1] - positions[0]
            image = torch.round(d @ torch.linalg.inv(cell)).detach()
            d = d - image @ cell
            return 0.5 * 10.0 * (torch.linalg.vector_norm(d) - 1.0) ** 2

    model = HarmonicPair()
    # aot_eager exercises graph capture and autograd compilation without requiring
    # a platform-specific optimizing compiler. This is not a speed benchmark.
    compiled_model = torch.compile(model, backend="aot_eager", fullgraph=True)

    def evaluate(system, kernel=model):
        positions = system.positions.detach().clone().requires_grad_(True)
        cell = system.cell.detach().clone().requires_grad_(True)
        energy = kernel(positions, cell)
        dE_dpositions, dE_dcell = torch.autograd.grad(energy, (positions, cell))
        strain_gradient = positions.T @ dE_dpositions + cell.T @ dE_dcell
        stress = (strain_gradient + strain_gradient.T) / (
            2 * torch.abs(torch.linalg.det(cell))
        )
        return energy.detach(), -dE_dpositions.detach(), stress.detach()

    mo.show_code(position="above")
    return compiled_model, evaluate


@app.cell
def _(mo):
    mo.md(r"""
    ## Choose a separation
    Substitute a numeric separation in a plain script.
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
    ## Compare eager and compiled predictions
    The assertions compare energy, forces, and stress, rather than checking energy alone.
    """)
    return


@app.cell
def _(System, compiled_model, evaluate, mo, separation, torch):
    def make_system(distance):
        return System(
            "angstrom",
            torch.tensor([1, 1], dtype=torch.int32),
            torch.tensor(
                [[1.0, 1.0, 1.0], [1.0 + distance, 1.0, 1.0]], dtype=torch.float64
            ),
            5.0 * torch.eye(3, dtype=torch.float64),
            torch.ones(3, dtype=torch.bool),
            arrays_backend="torch",
        )

    system = make_system(separation.value)
    energy, forces, stress = evaluate(system)
    compiled_results = evaluate(system, compiled_model)
    for _actual, _compiled in zip((energy, forces, stress), compiled_results):
        torch.testing.assert_close(_actual, _compiled)
    _display = mo.vstack(
        [
            mo.md(
                f"**E = {float(energy):.6f} eV**; eager and compiled results agree. Forces: eV/Å; stress: eV/Å³."
            ),
            forces,
            stress,
        ]
    )
    mo.show_code(_display, position="above")
    return energy, forces, stress, system


@app.cell
def _(mo):
    from metatomic_marimo.visualization import structure_panel

    mo.show_code(position="above")
    return (structure_panel,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Inspect the prediction
    The plotted values are detached only after derivatives have been computed.
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
    `fullgraph=True` checks the tensor kernel's graph capture; it does not establish
    that the ctypes-backed `System` wrapper or a future `TorchModel` can be compiled.
    Existing `metatomic.torch` models use a separate interface and need a dedicated
    adapter/compatibility tutorial once that contract is available.

    [PyTorch DLPack documentation](https://docs.pytorch.org/docs/stable/generated/torch.from_dlpack.html)
    · [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise
    Compare the same input twice with eager and compiled kernels. Then try starting autograd on a borrowed System tensor without keeping a reference: why should the evaluator instead extract each tensor once?


    **Worked answer.** A borrowed tensor is a storage view, not a reliable carrier of an earlier graph. Retain each extracted tensor, clone it into a fresh leaf, and differentiate the energy evaluated from those leaves. Repeat calls should give identical predictions.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce and continue
    Run `uv run marimo edit notebooks/04_torch_energy_forces_stress.py` from this project. Use the default controls to reproduce the reference result, then vary one input at a time. Replace `.value` controls with numeric values when copying to a script. Computation cells show their executed source; plotting helpers belong to this cookbook.

    This edition targets core System revision `b02b9ff3`, Python 3.12 and CPU float64. Source and dependencies are pinned in the guide; HTML exports preserve results but cannot rerun Python controls.

    [Torch autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html) · [Guide](/guide) · [API status](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
