import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 09 · Batch evaluations and measure the work you actually run

    **Performance · Executable tutorial**

    **Learning objective.** Separate batch correctness, compilation, synchronized timing and profiler interpretation.

    **Prerequisites.** 03–04; batch axes.


    This extends the themes of the published
    [profiling](https://docs.metatensor.org/metatomic/latest/examples/4-profiling.html)
    and [batched TorchSim](https://docs.metatensor.org/metatomic/latest/examples/6-torchsim-batched.html)
    tutorials. We evaluate equal-size core geometries on CPU; we do not claim to
    implement TorchSim's variable-size simulation state or its model adapter.

    **Expected result.** 24 equal-size systems agree with a scalar loop and a Torch implementation; wall-clock timings vary by machine.

    ## Construct a batch explicitly
    Each example has two atoms and a 5 Å cell. We batch arrays from core Systems,
    not Python wrappers carrying attached data. Different atom counts need padding
    and masks or a packed representation with system indices.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import jax
    import jax.numpy as jnp
    import torch
    import time
    from metatomic import System

    mo.show_code(jax.config.update("jax_enable_x64", True), position="above")
    return System, jax, jnp, mo, np, time, torch


@app.cell
def _(System, jnp, mo, np):
    distances = np.linspace(0.85, 1.55, 24)
    systems = [
        System(
            "angstrom",
            np.array([1, 1], dtype=np.int32),
            np.array([[1.0, 1.0, 1.0], [1.0 + r, 1.0, 1.0]]),
            5 * np.eye(3),
            np.ones(3, dtype=bool),
        )
        for r in distances
    ]
    positions = jnp.array(np.stack([s.positions for s in systems]))
    cells = jnp.array(np.stack([s.cell for s in systems]))
    mo.show_code(position="above")
    return cells, distances, positions


@app.cell
def _(mo):
    mo.md(r"""
    ## Differentiate one sample, then vectorize
    `vmap` maps the single-system function over both leading batch axes. Check against
    an explicit Python loop before using performance numbers. These results contain
    forces; add the cell/strain derivative from notebook 03 when stress is needed.
    """)
    return


@app.cell
def _(cells, distances, jax, jnp, mo, np, positions):
    def single_energy(r, c):
        d = r[1] - r[0]
        d = d - jnp.round(d @ jnp.linalg.inv(c)) @ c
        return 5 * (jnp.linalg.norm(d) - 1) ** 2

    single = jax.value_and_grad(single_energy, argnums=0)
    batched = jax.jit(jax.vmap(single, in_axes=(0, 0)))
    batch_energies, batch_gradients = batched(positions, cells)
    loop_results = [single(r, c) for r, c in zip(positions, cells, strict=True)]
    np.testing.assert_allclose(
        batch_energies, np.array([e for e, g in loop_results]), atol=1e-12
    )
    np.testing.assert_allclose(
        batch_gradients, np.array([g for e, g in loop_results]), atol=1e-12
    )
    _display = mo.ui.table(
        [
            {
                "distance": float(r),
                "energy": float(e),
                "force_atom_0_x": float(-g[0, 0]),
            }
            for r, e, g in zip(distances, batch_energies, batch_gradients, strict=True)
        ],
        selection=None,
    )
    mo.show_code(_display, position="above")
    return batch_energies, batch_gradients, batched


@app.cell
def _(mo):
    mo.md(r"""
    ## Warm up, synchronize, then time repeated execution
    JAX dispatch may be asynchronous. Wait on **all** returned arrays. Exclude initial
    compilation from steady-state timing, and report shapes/device/repetition count.
    These tiny CPU measurements describe this session, not a general JAX/Torch ranking.
    """)
    return


@app.cell
def _(batched, cells, jax, mo, np, positions, time):
    jax.block_until_ready(batched(positions, cells))
    repeats, calls_per_repeat = 7, 10
    timings = []
    for _repeat in range(repeats):
        start = time.perf_counter()
        for _call in range(calls_per_repeat):
            jax.block_until_ready(batched(positions, cells))
        timings.append((time.perf_counter() - start) / calls_per_repeat)
    median_ms = 1000 * np.median(timings)
    _display = mo.md(
        f"24 systems; float64; {jax.devices()[0]}; {repeats} repeats of "
        f"{calls_per_repeat} synchronized calls. Median warm batch: **{median_ms:.3f} ms** "
        f"(range {1000 * min(timings):.3f}–{1000 * max(timings):.3f} ms). "
        "This measures prepared-array energy and gradient evaluation, including Python dispatch; "
        "geometry conversion and pair construction are excluded."
    )
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Give the Torch profiler a meaningful label
    The profiler records operator activity under your named region. Here positions
    are already prepared; label conversion/neighbor search separately in a real engine.
    CPU profiling adds overhead and is not a benchmark. GPU profiling also requires
    device activities and synchronization appropriate to the device.
    """)
    return


@app.cell
def _(batch_energies, batch_gradients, mo, np, positions, torch):
    tr = torch.tensor(np.array(positions), dtype=torch.float64, requires_grad=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        with torch.profiler.record_function(
            "tutorial::batched_harmonic_energy_and_forces"
        ):
            d = tr[:, 1] - tr[:, 0]
            energies_torch = 5 * (torch.linalg.vector_norm(d, dim=1) - 1) ** 2
            gradients_torch = torch.autograd.grad(energies_torch.sum(), tr)[0]
    np.testing.assert_allclose(
        energies_torch.detach().numpy(), batch_energies, atol=1e-12
    )
    np.testing.assert_allclose(
        gradients_torch.detach().numpy(), batch_gradients, atol=1e-12
    )
    profile_report = profile.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=8
    )
    _display = mo.md("```text" + chr(10) + profile_report + chr(10) + "```")
    mo.show_code(_display, position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""
    The Torch kernel above omits periodic wrapping because this batch stays inside
    its nearest image. For arbitrary geometry, use the periodic kernel in notebook 04.

    **Try it:** increase the batch size and measure again after recompilation. Then
    profile the complete engine boundary. Do not count compilation as steady-state
    execution or compare a force calculation with an energy-only calculation.

    The published [TorchSim introduction](https://docs.metatensor.org/metatomic/latest/examples/5-torchsim-getting-started.html)
    remains the reference for its existing adapter. Support for the new core model
    base classes depends on that adapter's future contract; `vmap` alone does not
    provide a TorchSim engine integration.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding
    What must stay fixed when comparing a scalar loop with a batched evaluator?

    **Worked answer.** A speed comparison must hold outputs, dtype, device and batch size fixed. Report compile time separately and synchronize all outputs. Profiling operator time is not the same measurement as end-to-end throughput.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce and continue
    Run `uv run marimo edit notebooks/09_batching_and_profiling.py` from this project. Use the default controls to reproduce the reference result, then vary one input at a time. Replace `.value` controls with numeric values when copying to a script. Computation cells show their executed source; plotting helpers belong to this cookbook.

    This edition targets core System revision `b02b9ff3`, Python 3.12 and CPU float64. Source and dependencies are pinned in the guide; HTML exports preserve results but cannot rerun Python controls.

    [Published profiling tutorial](https://docs.metatensor.org/metatomic/latest/examples/4-profiling.html) · [Guide](/guide) · [API status](/roadmap)
    """)
    return


if __name__ == "__main__":
    app.run()
