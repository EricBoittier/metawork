import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 20 · Differentiate and compile a Torch LJ kernel

    **Reference model · Executable tutorial**

    **Learning objective.** Implement the accepted energy-shifted potential as an `nn.Module`, compare autograd with an analytic derivative, and validate graph capture separately from plugin execution.

    **Prerequisites.** 04 and 19.

    The numerical target is the same CPU float64 potential as chapter 19. This lesson uses a normal Torch module. It does not implement the pending public `TorchModel` or compile the ctypes-backed System boundary.


    **Expected result.** Eager, `aot_eager` and NumPy derivative values agree within float64 tolerance; the local labeled Torch adapter also agrees with the native plugin.

    ## Keep state and tensor operations explicit
    Fixed parameters are registered buffers. Parameters intended for fitting would need an explicit training contract. Pass pair vectors as tensor inputs, with discrete topology already resolved by the engine.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    from metatomic import Quantity
    from metatomic_marimo.lj_reference import (
        NativeLJ,
        NumpyLJ,
        TorchLJ,
        make_pair_system,
    )

    mo.show_code(position="above")
    return (
        NativeLJ,
        NumpyLJ,
        Quantity,
        TorchLJ,
        make_pair_system,
        mo,
        np,
        torch,
    )


@app.cell
def _(NumpyLJ, mo, np, torch):
    class ShiftedLJ(torch.nn.Module):
        def __init__(self, sigma=1.0, epsilon=1.0, cutoff=3.0):
            super().__init__()
            self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float64))
            self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float64))
            self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float64))

        def forward(self, vectors):
            squared = (vectors * vectors).sum(dim=1)
            sixth = (self.sigma**2 / squared) ** 3
            cutoff_sixth = (self.sigma / self.cutoff) ** 6
            energy = (
                4 * self.epsilon * (sixth**2 - sixth - cutoff_sixth**2 + cutoff_sixth)
            )
            return torch.where(squared < self.cutoff**2, energy, 0.0)

    def value_gradient(kernel, vectors):
        leaf = vectors.detach().clone().requires_grad_(True)
        energy = kernel(leaf)
        gradient = torch.autograd.grad(energy.sum(), leaf)[0]
        return energy.detach(), gradient.detach()

    model = ShiftedLJ()
    vectors = torch.tensor(
        [[1.2, 0.2, 0.0], [2.0, 0.0, 0.0], [3.2, 0.0, 0.0]], dtype=torch.float64
    )
    eager = value_gradient(model, vectors)
    compiled_kernel = torch.compile(model, backend="aot_eager", fullgraph=True)
    compiled = value_gradient(compiled_kernel, vectors)
    analytic = NumpyLJ().pair_energy_gradient(vectors.numpy())
    for actual, expected, reference in zip(compiled, eager, analytic, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.numpy(), reference, atol=1e-11)
    mo.show_code(eager, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    `aot_eager` checks graph capture and autograd behavior; it does not demonstrate an optimized speedup. Derivatives are tested away from the cutoff. The mask is a physical convention, not a differentiable neighbor-list algorithm.

    ## Verify the model-to-engine output boundary
    The local `TorchLJ` reference converts pair derivatives into labeled position gradients. The CPU conversion is deliberate for this comparison; a production GPU integration needs a separate device and ownership design.
    """)
    return


@app.cell
def _(NativeLJ, Quantity, TorchLJ, make_pair_system, mo, np):
    systems = [make_pair_system(1.2, transverse=0.2, image=True), make_pair_system(2.0)]
    requests = [
        Quantity(
            name="energy", unit="eV", sample_kind="system", gradients=["positions"]
        )
    ]
    torch_outputs = TorchLJ().execute(systems, None, requests)
    with NativeLJ() as native:
        cpp_outputs = native.execute(systems, None, requests)
    np.testing.assert_allclose(
        torch_outputs[0].block().values, cpp_outputs[0].block().values, atol=1e-12
    )
    np.testing.assert_allclose(
        torch_outputs[0].block().gradient("positions").values,
        cpp_outputs[0].block().gradient("positions").values,
        atol=1e-11,
    )
    mo.show_code(torch_outputs[0].block().values, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    Why is an energy-only compiled/eager comparison insufficient?

    **Worked answer.** A model can return the correct scalar while disconnecting geometry from its computation graph. Check forces and cell/strain derivatives when supported, not just energy. This LJ reference advertises position gradients only; stress requires an explicit extension and validation.

    ## Reproduce and references
    Run `uv run marimo edit notebooks/20_lj_torch.py`. Compilation occurs on the first evaluation; no throughput claim is made. Native execution requires the plugin in the pinned environment.

    [Accepted LJ implementation](https://github.com/metatensor/metatomic/blob/12b24b14/metatomic-core/lj-plugin/lennard_jones.cpp) · [Torch autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html) · [Guide](/guide)
    """)
    return


if __name__ == "__main__":
    app.run()
