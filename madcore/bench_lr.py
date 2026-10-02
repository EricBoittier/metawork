"""Time LOREM training steps (energy + forces, double backward) on real batches.

Batches are drawn like the trainer's (<= 64 structures, <= 1024 atoms) from a
DiskDataset zip with a fixed seed, so two code trees see identical batches.
Run with PYTHONPATH pointing at the metatrain tree to measure.

    python bench_lr.py train.zip --long-range true --steps 40 [--profile]
"""

import argparse
import random
import time

import torch
from metatomic.torch import ModelOutput

from metatrain.experimental.lorem import LOREM
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo, DiskDataset
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

p = argparse.ArgumentParser()
p.add_argument("zip")
p.add_argument("--long-range", default="true")
p.add_argument("--steps", type=int, default=40)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--max-atoms", type=int, default=1024)
p.add_argument("--profile", action="store_true")
args = p.parse_args()

torch.manual_seed(0)
dev = torch.device("cuda")
data = DiskDataset(args.zip)
rng = random.Random(0)
order = list(range(len(data)))
rng.shuffle(order)

hypers = get_default_hypers("experimental.lorem")["model"]
hypers["long_range"] = args.long_range == "true"
target = get_energy_target_info("energy", {"quantity": "energy", "unit": "eV"}, add_position_gradients=True)


def batches():
    it = iter(order)
    while True:
        systems, atoms = [], 0
        for idx in it:
            s = data[idx].system
            if systems and (len(systems) == args.batch_size or atoms + len(s) > args.max_atoms):
                yield systems
                systems, atoms = [], 0
            systems.append(s)
            atoms += len(s)


n_needed = args.warmup + args.steps
raw = []
for b in batches():
    raw.append(b)
    if len(raw) == n_needed:
        break
types = sorted({int(t) for b in raw for s in b for t in s.types})
# all elements up to Z=102 (MAD has nobelium) so every tree builds the same embedding
model = LOREM(hypers, DatasetInfo(length_unit="angstrom", atomic_types=list(range(1, 103)), targets={"energy": target})).to(dev)
nl = model.requested_neighbor_lists()
prepared = [[get_system_with_neighbor_lists(s.to(torch.float32), nl).to(dev) for s in b] for b in raw]
n_struct = sum(len(b) for b in prepared[args.warmup:])
n_atoms = sum(len(s) for b in prepared[args.warmup:] for s in b)


def step(systems):
    for s in systems:
        s.positions.requires_grad_(True)
    e = model(systems, {"energy": ModelOutput(sample_kind="system")})["energy"].block().values
    grads = torch.autograd.grad(e.sum(), [s.positions for s in systems], create_graph=True)
    loss = (e**2).sum() + sum((g**2).sum() for g in grads)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return loss


for b in prepared[: args.warmup]:
    step(b)
torch.cuda.synchronize()
t0 = time.perf_counter()
for b in prepared[args.warmup:]:
    step(b)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"long_range={hypers['long_range']} steps={args.steps} structures={n_struct} atoms={n_atoms}")
print(f"ms/step {1e3 * dt / args.steps:.1f}  ms/structure {1e3 * dt / n_struct:.3f}  us/atom {1e6 * dt / n_atoms:.1f}")
print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

if args.profile:
    from torch.profiler import ProfilerActivity, profile, record_function

    lr = model.lr
    if lr is not None:
        inner = lr._potentials

        def wrapped(*a, **k):
            with record_function("LR::potentials"):
                return inner(*a, **k)

        lr._potentials = wrapped
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for b in prepared[args.warmup : args.warmup + 10]:
            with record_function("STEP"):
                step(b)
        torch.cuda.synchronize()
    ka = prof.key_averages()
    for key in ("STEP", "LR::potentials"):
        ev = [e for e in ka if e.key == key]
        if ev:
            print(f"{key}: CPU total {ev[0].cpu_time_total / 1e3 / 10:.1f} ms/step, calls {ev[0].count}")
    print(ka.table(sort_by="cpu_time_total", row_limit=25, max_name_column_width=60))
    n_kernels = sum(e.count for e in ka if e.device_type.name == "CUDA")
    print(f"CUDA kernel launches per step: {n_kernels / 10:.0f}")
