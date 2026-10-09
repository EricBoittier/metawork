import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 17 · TorchSim: packed systems and cell orientation

    **Integration design note · Executable boundary fixture**

    **Learning objective.** Check the packed atom-to-system map and transpose cell vectors at an explicit boundary.

    **Prerequisites.** 04 and 09.

    **Scope.** Python execution and Torch contracts; TorchSim is not installed or executed here. The migration remains future work. The calculation below is complete and reproducible independently of the engine.

    ## Physical and data contract

    The published SimState reference uses a leading system axis for cells and `system_idx` to associate packed atoms with a system. It documents column-vector cells; core System uses row-vector cells. A cubic cell cannot reveal an accidental transpose, so the fixture uses an off-diagonal cell.

    This exercise uses plain Torch tensors to test the conversion logic. It does not construct a SimState or claim adapter compatibility.
    """)
    return


@app.cell
def _():
    import marimo as mo

    mo.show_code(position="above")
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Run the boundary fixture
    Read the dimensional or indexing argument first, then run the assertion. Passing this calculation validates only the stated boundary property.
    """)
    return


@app.cell
def _(mo):
    import torch

    core_cells = torch.tensor(
        [
            [[4.0, 0.0, 0.0], [0.7, 5.0, 0.0], [0.2, 0.3, 6.0]],
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        ],
        dtype=torch.float64,
    )
    engine_cells = core_cells.transpose(-1, -2).contiguous()
    fractional = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)
    cartesian_row = fractional @ core_cells[0]
    cartesian_column = engine_cells[0] @ fractional
    assert torch.allclose(cartesian_row, cartesian_column)
    system_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)
    atomic_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    system_values = torch.zeros(2, dtype=torch.float64).index_add(
        0, system_idx, atomic_values
    )
    assert torch.equal(system_values, torch.tensor([3.0, 12.0], dtype=torch.float64))
    mo.show_code(system_values, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Migration specification

    | Boundary | Decision required | Test |
    |---|---|---|
    | Packed state | Atom/system indexing and cell layout | Unequal atom counts; off-diagonal cells |
    | Model request | Quantity labels, gradients and unsupported outputs | Single/batched parity |
    | Autograd | Retained graph versus explicit gradients | Finite differences; graph-boundary checks |
    | Compilation | Tensor regions and dynamic topology | Eager/compiled outputs on supported shapes |
    | Device | Storage ownership, copies and dtype | CPU/device parity where available |

    Begin with one supported model and eager execution. Add heterogeneous batches before making throughput claims. GPU and compiler support are separate acceptance targets.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    Why is testing only cubic cells insufficient?

    **Worked answer.** A diagonal cell equals its transpose, so both conventions accidentally give the same coordinates. An off-diagonal cell makes the boundary transformation observable.

    ## Reproduce and references
    Run `uv run marimo edit notebooks/17_torchsim.py` in the pinned cookbook environment. The fixture uses CPU arrays; no engine build is required. For the full model contract and known native limitations, read [21](/notebooks/cpp-lj/).

    [TorchSim SimState reference](https://torchsim.github.io/torch-sim/reference/torch_sim.state.SimState.html) · [API roadmap](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
