---
tags: [ml, distributed, ddp, pytorch]
---
# DDP (DistributedDataParallel)

Every rank has a full model copy; gradients are **all-reduced** (averaged) during `backward()` in buckets, overlapping comm with compute.

## Minimal recipe
```python
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# init_process_group — see [[Distributed overview]]
model = DDP(model.to(local), device_ids=[local])
sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
loader = DataLoader(ds, batch_size=per_rank_bs, sampler=sampler, num_workers=8, pin_memory=True)

for epoch in range(n_epochs):
    sampler.set_epoch(epoch)              # otherwise every epoch has the same shuffle
    for batch in loader:
        loss = loss_fn(model(batch), batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()                   # all-reduce happens here
        opt.step()

if rank == 0:
    torch.save(model.module.state_dict(), "ckpt.pt")   # .module = unwrapped model
dist.destroy_process_group()
```

## Rules
- Only rank 0 logs / writes checkpoints / prints; others `dist.barrier()` if they must wait.
- Metrics: all-reduce before logging, or you only see rank 0's shard:
  `dist.all_reduce(t, op=dist.ReduceOp.SUM); t /= world`.
- Same seed for model init on all ranks (or rely on DDP's initial broadcast from rank 0, which it does).
- Different data on each rank (sampler), same everything else.

## Gradient accumulation
```python
for i, batch in enumerate(loader):
    ctx = model.no_sync() if (i + 1) % accum != 0 else contextlib.nullcontext()
    with ctx:
        (loss_fn(model(batch), batch) / accum).backward()
    if (i + 1) % accum == 0:
        opt.step(); opt.zero_grad(set_to_none=True)
```

## Force training (double backward)
- Forces via `autograd.grad(E, pos, create_graph=True)` inside forward is fine with DDP.
- If some params get no grad on some steps → `find_unused_parameters=True` (slower) — better to make the graph static.
- `static_graph=True` can help if the graph never changes.

## Variable-size batches (atoms per graph differ)
- Ranks do unequal work → fast ranks wait at the all-reduce. Batch by **atom count** not structure count, or use `model.join()` context for uneven final batches.

## Tuning
- `bucket_cap_mb` (default 25): bigger buckets = fewer, larger all-reduces.
- `gradient_as_bucket_view=True` saves a copy.
- Mixed precision comm hook: `model.register_comm_hook(None, default_hooks.bf16_compress_hook)`.
- Combine with `torch.compile(model)` — compile **after** wrapping with DDP.

Related: [[FSDP]], [[Scaling]], [[Benchmarking and profiling]]
