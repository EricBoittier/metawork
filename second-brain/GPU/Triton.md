---
tags: [gpu, triton]
---
# Triton

Python DSL for GPU kernels; works on **blocks** not threads. Ships with PyTorch (`torch.compile` generates Triton). Version is tied to torch — don't `pip install triton` separately into a torch venv.

## Skeleton
```python
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK"]),)
    add_kernel[grid](x, y, out, x.numel(), BLOCK=1024)
    return out
```

## Autotune
```python
@triton.autotune(
    configs=[triton.Config({"BLOCK": b}, num_warps=w) for b in (256, 512, 1024) for w in (4, 8)],
    key=["n"],
)
@triton.jit
def kernel(...): ...
```

## Benchmark
```python
ms = triton.testing.do_bench(lambda: add(x, y))     # handles warmup + sync, returns ms
gbps = 3 * x.numel() * x.element_size() / ms * 1e-6
```
Compare against the torch baseline and the HBM roofline — see [[Benchmarking and profiling]].

## Debug
```bash
TRITON_INTERPRET=1 python k.py          # run kernel on CPU in numpy; print() works
TRITON_PRINT_AUTOTUNING=1 python k.py
TRITON_CACHE_DIR=/tmp/tc python k.py    # isolate/clear compile cache (~/.triton/cache)
```
`tl.device_print("x", x)` inside kernels on GPU.

## Inspect what torch.compile generated
```bash
TORCH_LOGS=output_code python train.py
TORCH_COMPILE_DEBUG=1 python train.py   # dumps to ./torch_compile_debug/
```

## Gotchas
- `tl.constexpr` args recompile per distinct value.
- Block sizes must be powers of 2 for `tl.arange`.
- Reductions/atomics: `tl.atomic_add` exists, but sorted segment-sums are often faster.
- Blackwell (`sm_100/120`) needs a recent Triton (bundled with torch ≥ 2.7).

Related: [[CUDA]], [[PyTorch]]
