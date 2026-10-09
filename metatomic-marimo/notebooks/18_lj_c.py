import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 18 · Load and execute the C++ LJ reference

    **Reference model · Executable tutorial**

    **Learning objective.** Inspect a native model’s capabilities, satisfy its pair request and convert its energy gradient to forces.

    **Prerequisites.** 00, 01 and 07. The pinned environment must contain the installed `metatomic-lj-plugin.so` artifact.

    The accepted model implements a C ABI plugin in C++. It is a reference for engine development, not a fitted material model. In this lesson $\sigma=1$ Å, $\epsilon=1$ eV and $r_c=3$ Å. Its potential is energy-shifted below cutoff and zero at or beyond cutoff:
    $$U(r)=4\epsilon[(\sigma/r)^{12}-(\sigma/r)^6]-U_{\rm LJ}(r_c).$$


    **Expected result.** At $r=2$ Å, energy is approximately −0.056044 eV. Atomic forces are equal and opposite. The model advertises position gradients, not strain gradients.

    ## Define the request before executing
    The loader options are strings. The model requests a non-strict half list; an engine can include skin pairs, but the model must filter them. `Quantity` describes the requested output and its explicit gradients.
    """)
    return


@app.cell
def _():
    import inspect
    import json
    import marimo as mo
    import numpy as np
    from metatomic import Quantity
    from metatomic_marimo.lj_reference import (
        DEFAULT_OPTIONS,
        NativeLJ,
        make_pair_system,
    )
    from metatomic_marimo.visualization import structure_panel

    mo.show_code(position="above")
    return (
        DEFAULT_OPTIONS,
        NativeLJ,
        Quantity,
        inspect,
        json,
        make_pair_system,
        mo,
        np,
        structure_panel,
    )


@app.cell
def _(Quantity, mo):
    request = Quantity(
        name="energy", unit="eV", sample_kind="system", gradients=["positions"]
    )
    separation = mo.ui.slider(
        1.05, 3.2, step=0.05, value=2.0, label="Pair separation (Å)"
    )
    mo.show_code(separation, position="above")
    return request, separation


@app.cell
def _(mo):
    mo.md(r"""
    ## Own the native handle explicitly
    `NativeLJ` is a cookbook-only bridge through private Python bindings to the public C API. Its source is shown below. It is not a shipped `metatomic.load_model` function. `with` releases the model once; returned TensorMaps retain independent output ownership. The helper registers a CPU array wrapper for the C++ data origin.
    """)
    return


@app.cell
def _(
    DEFAULT_OPTIONS,
    NativeLJ,
    json,
    make_pair_system,
    mo,
    np,
    request,
    separation,
):
    system = make_pair_system(separation.value)
    with NativeLJ(DEFAULT_OPTIONS) as native_model:
        capabilities = native_model.capabilities()
        requested_pairs = native_model.requested_pair_lists()
        assert requested_pairs == system.known_pairs()
        outputs = native_model.execute([system], None, [request])
    block = outputs[0].block()
    energy = float(block.values[0, 0])
    forces = -np.array(block.gradient("positions").values[:, :, 0])
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-12)
    _display = mo.vstack(
        [
            mo.md(
                f"Energy: **{energy:.8f} eV**. Samples: `{block.samples.names}`; gradient samples: `{block.gradient('positions').samples.names}`."
            ),
            forces,
            mo.md("```json\n" + json.dumps(capabilities.to_dict(), indent=2) + "\n```"),
        ]
    )
    mo.show_code(_display, position="above")
    return energy, forces, system


@app.cell
def _(energy, forces, mo, structure_panel, system):
    _display = structure_panel(system, energy=energy, forces=forces)
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    The fixture provides one known pair and is suitable for this large, dilute periodic cell. It is not a neighbor-search implementation; chapter 06 constructs actual periodic topology. At and above cutoff the included skin pair contributes zero. The force has a jump at cutoff because only the energy has been shifted.
    """)
    return


@app.cell
def _(NativeLJ, inspect, mo):
    _display = mo.accordion(
        {
            "Native bridge source": mo.md(
                "```python" + chr(10) + inspect.getsource(NativeLJ) + chr(10) + "```"
            )
        }
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    Why is the force not zero just below the cutoff even though the energy approaches zero?

    **Worked answer.** Subtracting a constant changes the energy reference but not its radial derivative. Force shifting would also subtract a linear term; that is a different potential. Do not change conventions while comparing implementations.

    ## Validated scope and limitations
    The suite checks native/NumPy/Torch/JAX values and gradient labels, cutoff cases, periodic images, selections spanning all input systems, empty requests and output lifetime after close. Checked native system-energy execution currently rejects a selection omitting an entire system; chapter 21 records that limitation. No native stress, GPU or MPI claim is made.

    ## Reproduce and references
    Run `uv run marimo edit notebooks/18_lj_c.py`. The default result uses the source revision and environment pinned in the guide.

    [Accepted LJ implementation](https://github.com/metatensor/metatomic/blob/12b24b14/metatomic-core/lj-plugin/lennard_jones.cpp) · [Cross-backend comparison](/notebooks/cpp-lj/) · [Guide](/guide)
    """)
    return


if __name__ == "__main__":
    app.run()
