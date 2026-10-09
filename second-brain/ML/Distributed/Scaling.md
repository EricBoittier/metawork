---
tags: [ml, distributed, scaling, hpc]
---
# Scaling

## Two experiments, always both
| | Fix | Vary | Ideal |
|---|---|---|---|
| **Strong scaling** | total problem (global batch / system size) | GPUs | time ∝ 1/N |
| **Weak scaling** | work per GPU (per-rank batch) | GPUs | time constant |

- Speedup `S(N) = T(1) / T(N)`; efficiency `E(N) = S(N) / N`. Report E, not just S. >80% is good, <50% means stop adding GPUs.
- Amdahl: serial fraction `s` caps speedup at `1/s` — e.g. 5% serial (data loading, logging, rank-0 work) → max 20×.
- Plot on log–log with the ideal line; x-axis GPUs (1, 2, 4, 8, 16…).

## Where scaling breaks
1. **Communication**: all-reduce time ~ params × 2(N−1)/N / bandwidth. Intra-node NVLink ≫ inter-node network → expect a drop at the first multi-node point. Check `nvidia-smi topo -m`.
2. **Load imbalance**: variable atoms per structure → slowest rank sets the pace. Bucket/balance by atom count.
3. **Data pipeline**: CPU workers per GPU (`--cpus-per-task`), shared filesystem I/O. Pre-shard / cache datasets on node-local scratch.
4. **Small per-GPU work**: kernels too small to saturate the GPU → weak scaling looks fine, strong scaling collapses.

## Batch size and learning rate
- Growing world_size grows the **global batch**. Either keep global batch fixed (per-rank batch ↓) or rescale LR.
- Linear scaling rule (SGD-ish): LR ∝ global batch, with warmup. Adam: √ scaling is often safer. Always re-check validation curves, not just throughput.
- Report both **throughput** (structures/s or atoms/s) and **time-to-accuracy**.

## Multi-node on SLURM
```bash
#SBATCH --nodes=2 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=8 --qos=normal
export NCCL_DEBUG=WARN
srun --cpu-bind=cores python train.py
```
Measure collective bandwidth first:
```bash
# nccl-tests
srun ./build/all_reduce_perf -b 8M -e 1G -f 2 -g 1
```
Compare "busbw" with the expected NVLink / InfiniBand numbers before blaming your code.

## MD / inference scaling
- Spatial decomposition (LAMMPS with metatomic): more ranks = smaller domains, but ghost atoms grow with the model's cutoff × message-passing layers.
- Throughput metric: ns/day or atom-steps/s; record system size, cutoff, neighbor-list time vs model time.

## Checklist for a scaling study
- [ ] same code commit + venv on all points (`hpc_run.py` manifest)
- [ ] warmup excluded, ≥3 repeats, report median + spread
- [ ] GPUs per node and node count recorded (topology matters)
- [ ] both strong and weak scaling
- [ ] efficiency plot with ideal line
- [ ] note where the per-step breakdown shifts (compute vs comm vs data)

Related: [[Benchmarking and profiling]], [[DDP]], [[FSDP]], [[Kuma and SLURM]]
