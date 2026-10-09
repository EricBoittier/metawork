---
tags: [ml, distributed]
---
# Distributed training overview

## Which strategy?
| Strategy | Each GPU holds | Use when |
|---|---|---|
| [[DDP]] (data parallel) | full model + grads + optimizer | model fits on one GPU — **default for MLIPs** |
| [[FSDP]] / ZeRO-3 | a shard of params/grads/optim | model or optimizer state doesn't fit |
| ZeRO-1/2 | full params, sharded optim (+grads) | optimizer state (Adam = 2× params) is the problem |
| Tensor parallel | slice of each layer | very wide layers (LLMs) |
| Pipeline parallel | a subset of layers | very deep models, multi-node |
| Spatial / domain decomposition | a region of the system | huge MD systems (LAMMPS-style), not training |

Most atomistic models are small (≤100M params) → DDP; the bottleneck is usually graph/neighbour-list work and data loading, not memory.

## Vocabulary
- **world_size** = total processes (usually = total GPUs); **rank** = global id; **local_rank** = GPU index on this node.
- **Backend**: `nccl` for GPU, `gloo` for CPU.
- **Global batch** = per-rank batch × world_size (× grad-accum steps).
- Collectives: all-reduce (DDP grads), all-gather (FSDP params), reduce-scatter (FSDP grads), broadcast.

## Launching on SLURM (kuma)
Two options:
1. **srun, one task per GPU** (what pipeline-bench does):
   ```bash
   #SBATCH --nodes=1 --ntasks-per-node=2 --gres=gpu:2 --cpus-per-task=8
   export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
   export MASTER_PORT=29500
   srun python train.py      # read SLURM_PROCID / SLURM_LOCALID / SLURM_NTASKS
   ```
2. **torchrun, one task per node**:
   ```bash
   #SBATCH --nodes=2 --ntasks-per-node=1 --gres=gpu:4
   srun torchrun --nnodes=$SLURM_NNODES --nproc-per-node=4 \
     --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:29500 train.py
   ```
   torchrun sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`.

Init that handles both:
```python
import os, torch.distributed as dist
rank  = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
torch.cuda.set_device(local)
dist.init_process_group("nccl", rank=rank, world_size=world)
```
Gotcha seen in pipeline-bench: `SLURM_NTASKS` leaking from an outer allocation into an inner `srun` → wrong world size. Pass `--ntasks` explicitly.

## Debugging hangs
```bash
export NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1      # fail instead of hanging forever
py-spy dump --pid <pid>                       # on each rank: who's stuck where
```
Usual causes: ranks taking different code paths (one skips a collective), uneven last batch, mismatched tensor shapes in all-reduce, firewall/wrong interface (`NCCL_SOCKET_IFNAME`).

Related: [[DDP]], [[FSDP]], [[Scaling]], [[Kuma and SLURM]]
