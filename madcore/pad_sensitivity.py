"""Batching dependence of metatrain PET gradients (float64): the loss
gradient of structures 0..n-1 of DATA/train as one batch, and as two halves
whose gradients are summed (each loss normalised by the whole batch, so the
sums agree exactly unless padding to the batch's longest neighbour list
changes the model). Per-tensor relative differences, worst first.

    python pad_sensitivity.py --model start.ckpt --data DIR [--n 16] [--conservative]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from metatomic.torch import ModelOutput, System
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

sys.path.insert(0, "/home/boittier/metawork/mlip.cpp/scripts")
from export_pytorch.export_pet_full import load_local_model  # noqa: E402
sys.path.insert(0, "/home/boittier/metawork/mlip.cpp/scripts/train")
from check_gradients import read_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--conservative", action="store_true", help="loss on -dE/dx instead of the direct head")
    args = ap.parse_args()
    split = read_split(Path(args.data) / "train")
    model = load_local_model(args.model).to(torch.float64)
    model.train()
    nl_options = model.requested_neighbor_lists()

    def systems(ids):
        out = []
        for s in ids:
            lo, hi = split["na"][s], split["na"][s + 1]
            cell = torch.tensor(split["c"][s], dtype=torch.float64)
            pos = torch.tensor(split["x"][lo:hi], dtype=torch.float64, requires_grad=args.conservative)
            out.append(get_system_with_neighbor_lists(
                System(types=torch.tensor(split["a"][lo:hi]), positions=pos, cell=cell, pbc=(cell != 0).any(dim=1)),
                nl_options))
        return out

    n_atoms_total = int(split["na"][args.n] - split["na"][0])

    def grads(batches):
        model.zero_grad()
        for ids in batches:
            sys_ = systems(ids)
            out = model(sys_, {"energy": ModelOutput(sample_kind="atom"),
                               "non_conservative_force": ModelOutput(sample_kind="atom")})
            e = out["energy"].block().values.reshape(-1)
            if args.conservative:
                f = -torch.cat(torch.autograd.grad(e.sum(), [s.positions for s in sys_], create_graph=True))
            else:
                f = out["non_conservative_force"].block().values.reshape(-1, 3)
            counts = [len(s) for s in sys_]
            e_pa = torch.stack([c.sum() / len(c) for c in torch.split(e, counts)])
            # fixed pseudo-targets (zero): only the batching may change the result
            loss = (e_pa ** 2).sum() / args.n + (f ** 2).sum() / (3 * n_atoms_total)
            loss.backward()
        return {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}

    ids = list(range(args.n))
    whole = grads([ids])
    halves = grads([ids[: args.n // 2], ids[args.n // 2:]])
    rows = []
    for k, g in whole.items():
        d = (halves[k] - g).norm().item()
        rows.append((d / max(g.norm().item(), 1e-300), k, g.norm().item()))
    rows.sort(reverse=True)
    print(f"{len(rows)} tensors, worst relative difference whole vs halves {rows[0][0]:.2e}"
          f" ({'conservative' if args.conservative else 'direct'} forces)")
    for r, k, gn in rows[:12]:
        print(f"  {r:.2e}  {k}  |g| {gn:.3e}")


if __name__ == "__main__":
    main()
