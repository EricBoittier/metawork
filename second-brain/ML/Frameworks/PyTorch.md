---
tags: [ml, pytorch]
---
# PyTorch

## Device / dtype hygiene
```python
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.zeros(n, 3, device=dev, dtype=torch.float64)   # create on device, don't .to() later
torch.set_default_dtype(torch.float64)                   # atomistic models often need fp64 forces
torch.backends.cuda.matmul.allow_tf32 = True             # big speedup on Ampere+, slight precision loss
```

## Autograd for forces / stress
```python
pos.requires_grad_(True)
E = model(pos).sum()
F = -torch.autograd.grad(E, pos, create_graph=training)[0]   # create_graph=True if loss uses forces
```
- `torch.no_grad()` for inference without forces; `torch.inference_mode()` is faster still but forbids later autograd.
- Double backward (force training) is the expensive bit — profile it separately.

## torch.compile
```python
model = torch.compile(model)                  # default mode
model = torch.compile(model, mode="max-autotune", dynamic=True)
```
- Variable atom counts → recompiles; use `dynamic=True` or `torch._dynamo.mark_dynamic(x, 0)`.
- Debug graph breaks: `TORCH_LOGS=graph_breaks,recompiles python train.py`.
- `torch._dynamo.explain(model)(inputs)` summarises breaks.
- Always time compile separately from steady-state (see [[Benchmarking and profiling]]).

## Memory
```python
torch.cuda.max_memory_allocated() / 2**30
torch.cuda.reset_peak_memory_stats()
torch.cuda.memory._record_memory_history(); ...; torch.cuda.memory._dump_snapshot("mem.pickle")  # view at pytorch.org/memory_viz
```
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation OOMs.

## Mixed precision
```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = loss_fn(model(x), y)
loss.backward()            # bf16 → no GradScaler needed; fp16 → use torch.amp.GradScaler
```
Keep energies/forces accumulation in fp32/fp64 for MLIPs.

## Data loading
`DataLoader(ds, batch_size, num_workers=8, pin_memory=True, persistent_workers=True)` + `x.to(dev, non_blocking=True)`.
If GPU util < ~80%, suspect the loader first.

## Reproducibility
```python
torch.manual_seed(0); np.random.seed(0); random.seed(0)
torch.use_deterministic_algorithms(True)   # + CUBLAS_WORKSPACE_CONFIG=:4096:8
```
`scatter_add`/`index_add_` on CUDA are nondeterministic (atomics).

Related: [[PyTorch extensions and TorchScript]], [[DDP]], [[FSDP]], [[Training practice]]
