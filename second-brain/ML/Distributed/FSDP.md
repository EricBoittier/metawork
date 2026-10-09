---
tags: [ml, distributed, fsdp, pytorch]
---
# FSDP (Fully Sharded Data Parallel)

Params, grads and optimizer state are **sharded** across ranks. Before a layer runs, its params are **all-gathered**; after backward, grads are **reduce-scattered**. Memory per GPU ≈ total / world_size (+ one layer's full params at a time).

Use when the model + Adam state (≈ 4× params in fp32 + activations) doesn't fit. For most MLIPs, [[DDP]] is simpler and faster.

## FSDP2 (`fully_shard`) — current API
```python
# torch ≥ 2.6:
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
# torch 2.4–2.5 (our shared .venv is 2.5.1):
# from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for block in model.blocks:              # shard each transformer/message-passing block
    fully_shard(block, mp_policy=mp)
fully_shard(model, mp_policy=mp)        # then the root
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)   # build optimizer AFTER sharding
```
Params become `DTensor`s; the training loop is unchanged.

## FSDP1 (legacy wrapper) — older code
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
model = FSDP(model, auto_wrap_policy=..., sharding_strategy=ShardingStrategy.FULL_SHARD,
             device_id=local, use_orig_params=True)
```
| `ShardingStrategy` | ≈ |
|---|---|
| `FULL_SHARD` | ZeRO-3 |
| `SHARD_GRAD_OP` | ZeRO-2 |
| `HYBRID_SHARD` | shard within node, replicate across nodes |
| `NO_SHARD` | DDP |

## Wrapping granularity
- Too coarse (only root) → whole model gathered at once, no memory win.
- Too fine (every Linear) → many small collectives, slow.
- Sweet spot: one unit per block/layer.

## Checkpointing
```python
import torch.distributed.checkpoint as dcp
dcp.save({"model": model.state_dict(), "optim": opt.state_dict()}, checkpoint_id="ckpt/step100")
dcp.load(state, checkpoint_id="ckpt/step100")     # works with a different world size
```
For a single-file export (e.g. TorchScript for metatomic): gather a full state dict via `torch.distributed.checkpoint.state_dict.get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))` on rank 0.

## Extra memory levers
- Activation checkpointing: `torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)` — trade compute for memory.
- CPU offload (`CPUOffloadPolicy`) — big memory save, big slowdown.
- `reshard_after_forward=False` on the last blocks — keep params gathered, less comm.

## Gotchas
- Optimizer must be created after `fully_shard`.
- Ops that touch `.data` or param identity break with DTensor.
- Double backward (force loss) + FSDP: works, but test numerics against DDP on a small model first.

Related: [[DDP]], [[Distributed overview]], [[Scaling]]
