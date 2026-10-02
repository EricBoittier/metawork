"""Time turning a collated (pinned) batch into device systems + targets.

    A: unpack_batch on the host, then batch_to                 (what trainers do)
    C: as A, neighbor-list Labels built with assume_unique=True
    B: move the flat batch tensors first, split and build on the device

    python bench_unpack.py train.zip [--batches 60]
"""

import argparse
import time

import torch
from metatensor.torch import Labels, TensorBlock
from metatomic.torch import NeighborListOptions, System

from metatrain.utils.data import CollateFn, DiskDataset, unpack_batch
from metatrain.utils.data.dataset import (
    NEIGHBOR_SAMPLE_NAMES,
    load_buffer,
    neighbor_list_metadata,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists_transform
from metatrain.utils.transfer import batch_to

p = argparse.ArgumentParser()
p.add_argument("zip")
p.add_argument("--batches", type=int, default=60)
args = p.parse_args()

dev, dtype = torch.device("cuda"), torch.float32
nl = NeighborListOptions(cutoff=5.0, full_list=True, strict=True)
data = DiskDataset(args.zip, fields=["system", "energy"]) if False else DiskDataset(args.zip)
collate = CollateFn(target_keys=["energy"], callables=[get_system_with_neighbor_lists_transform([nl])])
loader = torch.utils.data.DataLoader(
    data, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0),
    collate_fn=collate, num_workers=8, pin_memory=True,
)
batches = []
for b in loader:
    batches.append(b)
    if len(batches) == args.batches:
        break
print("pinned:", batches[0].positions.is_pinned(), "atoms/batch", sum(sum(b.atom_counts) for b in batches) / len(batches))


def unpack_assume_unique(batch):
    systems = [
        System(types=t, positions=x, cell=c, pbc=p)
        for t, x, c, p in zip(torch.split(batch.types, batch.atom_counts), torch.split(batch.positions, batch.atom_counts), batch.cells, batch.pbcs)
    ]
    components, properties = neighbor_list_metadata()
    for neighbors in batch.neighbor_lists:
        for system, values, samples in zip(systems, torch.split(neighbors.values, neighbors.pair_counts), torch.split(neighbors.samples, neighbors.pair_counts)):
            system.add_neighbor_list(neighbors.options, TensorBlock(values=values, samples=Labels(NEIGHBOR_SAMPLE_NAMES, samples, assume_unique=True), components=components, properties=properties))
    buffers = iter(torch.split(batch.blob, batch.target_sizes + batch.extra_sizes + batch.data_sizes))
    targets = {name: load_buffer(next(buffers)) for name in batch.target_names}
    return systems, targets, {}


def unpack_on_device(batch):
    to = lambda x, dt=None: x.to(device=dev, dtype=dt or x.dtype, non_blocking=True)
    positions, types, cells, pbcs = to(batch.positions, dtype), to(batch.types), to(batch.cells, dtype), to(batch.pbcs)
    systems = [
        System(types=t, positions=x, cell=c, pbc=p)
        for t, x, c, p in zip(torch.split(types, batch.atom_counts), torch.split(positions, batch.atom_counts), cells, pbcs)
    ]
    components, properties = neighbor_list_metadata()
    components, properties = [components[0].to(dev)], properties.to(dev)
    for neighbors in batch.neighbor_lists:
        values, samples = to(neighbors.values, dtype), to(neighbors.samples)
        for system, v, s in zip(systems, torch.split(values, neighbors.pair_counts), torch.split(samples, neighbors.pair_counts)):
            system.add_neighbor_list(neighbors.options, TensorBlock(values=v, samples=Labels(NEIGHBOR_SAMPLE_NAMES, s, assume_unique=True), components=components, properties=properties))
    buffers = iter(torch.split(batch.blob, batch.target_sizes + batch.extra_sizes + batch.data_sizes))
    targets = {name: load_buffer(next(buffers)).to(dtype=dtype, device=dev, non_blocking=True) for name in batch.target_names}
    return systems, targets, {}


variants = {
    "A unpack+batch_to": lambda b: batch_to(*unpack_batch(b), dtype=dtype, device=dev),
    "C assume_unique+batch_to": lambda b: batch_to(*unpack_assume_unique(b), dtype=dtype, device=dev),
    "B move flat, build on device": unpack_on_device,
}
for name, fn in list(variants.items()) * 2:  # second round is the measurement
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for b in batches:
        out = fn(b)
    torch.cuda.synchronize()
    print(f"{name:32s} {1e3 * (time.perf_counter() - t0) / len(batches):7.2f} ms/batch")

# same content either way
ref, new = variants["A unpack+batch_to"](batches[0]), unpack_on_device(batches[0])
for s1, s2 in zip(ref[0], new[0]):
    assert torch.equal(s1.positions, s2.positions) and torch.equal(s1.types, s2.types) and torch.equal(s1.cell, s2.cell)
    n1, n2 = s1.get_neighbor_list(nl), s2.get_neighbor_list(nl)
    assert torch.equal(n1.values, n2.values) and torch.equal(n1.samples.values, n2.samples.values)
assert all(torch.equal(ref[1][k].block().values, new[1][k].block().values) for k in ref[1])
print("identical systems, neighbor lists and targets")
