---
tags: [ml, benchmarking, profiling, gpu]
---
# Benchmarking and profiling

## Rules for a benchmark you can trust
1. **Warm up** (CUDA context, cuDNN autotune, `torch.compile`/`jit` compilation, caches). Discard the first iterations; report compile time separately.
2. **Synchronize**: GPU work is async. `torch.cuda.synchronize()` / `jax.block_until_ready()` before stopping the clock.
3. **Repeat**: ≥ 3 runs × many iterations; report **median and IQR** (or min for micro-benchmarks), never one number.
4. **Fix everything else**: same commit, venv, GPU model, node, input sizes, dtype, batch. Record it all (the `hpc_run.py` manifest does this).
5. **Isolate**: whole node or at least whole GPU; nothing else running (`nvidia-smi`). MIG slices and shared nodes give noisy numbers.
6. **Realistic inputs**: real system sizes / atom-count distributions, not just one toy structure.
7. **Meaningful units**: atoms/s, structures/s, ns/day, steps/s, plus peak memory. Compare against a roofline or a baseline.
8. **Sweep**, don't spot-check: batch size × system size × dtype × device. Plot it.

## PyTorch timing
```python
import torch.utils.benchmark as tb
t = tb.Timer("model(x)", globals={"model": model, "x": x})
print(t.blocked_autorange(min_run_time=2.0))     # handles warmup + sync, gives median/IQR
```
Manual:
```python
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
for _ in range(10): model(x)                      # warmup
torch.cuda.synchronize(); start.record()
for _ in range(100): model(x)
end.record(); torch.cuda.synchronize()
ms = start.elapsed_time(end) / 100
```
Triton: `triton.testing.do_bench(fn)`. JAX: see [[JAX]].

## Profiling — find out *why*
```python
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=schedule(wait=1, warmup=2, active=3),
             on_trace_ready=tensorboard_trace_handler("./prof"),
             record_shapes=True, profile_memory=True, with_stack=True) as p:
    for step in range(6):
        train_step(); p.step()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```
Open traces at `ui.perfetto.dev`. Label regions: `with torch.profiler.record_function("neighbor_list"):` (shows in Nsight as NVTX too).

System-level:
```bash
nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o rep python bench.py   # timeline: gaps = CPU/launch bound
ncu --set full -k regex:kernel_name python bench.py                  # one kernel in depth
py-spy record -o cpu.svg -- python bench.py                          # CPU-side Python hotspots
```

## Reading the result
| Symptom | Likely cause | Fix |
|---|---|---|
| GPU idle gaps in timeline | data loader / Python overhead | more workers, pin_memory, `torch.compile`, CUDA graphs |
| Many tiny kernels | launch-bound | fuse (`torch.compile`, Triton), bigger batches |
| One kernel dominates, low SM% | memory-bound | better layout, fewer passes, fusion |
| `cudaMemcpy` / `.item()` / `.cpu()` in loop | host sync | remove syncs from hot path |
| Time grows step to step | recompiles / memory fragmentation | `TORCH_LOGS=recompiles`, `expandable_segments` |

## In metawork
- `pipeline-bench/` — Snakemake matrix (variant × batch × world_size), `submit.sh [--submit]` on kuma, results in `results/…/*.json`.
- Always note GPU type (L40S vs H100 vs B200 changes conclusions — L40S has weak FP64).

Related: [[Scaling]], [[CUDA]], [[Triton]], [[PyTorch]]
