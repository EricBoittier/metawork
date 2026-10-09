---
tags: [ml, jax]
---
# JAX

Used for `lorem-jax` and parity checks (`etc/lorem-parity`). **Separate venv** — never install JAX into the torch `.venv` (CUDA lib clashes).
```bash
uv pip install "jax[cuda12]"
python -c "import jax; print(jax.devices())"
```

## Core transforms
```python
import jax, jax.numpy as jnp
f_fast  = jax.jit(f)
grad_f  = jax.grad(energy)                 # scalar → gradient (forces = -grad)
e, g    = jax.value_and_grad(energy)(pos)
batched = jax.vmap(f, in_axes=(0, None))   # vectorise over first arg
hess_v  = jax.jvp(grad_f, (x,), (v,))[1]   # Hessian-vector product
```
- **Pure functions only**: no in-place mutation → `x = x.at[i].set(v)`, `x.at[idx].add(v)` for scatter.
- Randomness is explicit: `key, sub = jax.random.split(key)`.
- Static args (shapes, flags): `jax.jit(f, static_argnames=("n_layers",))`.

## Shapes and recompilation
- Every new input **shape** recompiles → pad atoms/neighbours to fixed buckets for MD/training.
- Python control flow on traced values fails → `jax.lax.cond`, `lax.scan`, `lax.fori_loop`.
- `jax.debug.print("{x}", x=x)` inside jit; `jax.config.update("jax_disable_jit", True)` to debug.

## Precision
```python
jax.config.update("jax_enable_x64", True)   # default is float32!
```
For parity with torch fp64 models this must be set **before** creating arrays.

## Benchmarking
```python
f(x).block_until_ready()          # warmup + compile
t0 = time.perf_counter(); f(x).block_until_ready(); dt = time.perf_counter() - t0
```
JAX is async — without `block_until_ready()` you time the dispatch only. Report compile time separately.

## Memory
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (stop grabbing 75% of GPU) or `XLA_PYTHON_CLIENT_MEM_FRACTION=.5`.

## Multi-device
`jax.sharding` + `jax.jit(..., in_shardings=..., out_shardings=...)`; `jax.distributed.initialize()` for multi-node (reads SLURM env automatically).

## Profiling
```python
with jax.profiler.trace("/tmp/jax-trace"): f(x).block_until_ready()   # open in TensorBoard / Perfetto
```
`f.lower(x).compile().as_text()` shows the optimized HLO.

Related: [[PyTorch]], [[Benchmarking and profiling]]
