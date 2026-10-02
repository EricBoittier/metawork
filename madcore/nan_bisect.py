"""Which structures of a saved non-finite batch give NaN parameter gradients?

Loads nan_batch.pt (written by the nan-debug trainer: systems, weights, hypers,
atomic types), and for each structure computes E and F = -dE/dx (create_graph),
the loss E^2 + |F|^2 and its parameter gradients, with the saved weights and
with fresh ones. Prints the structures with a non-finite result or a pair
closer than 0.5 A.

    python nan_bisect.py nan_batch.pt
"""
import sys

import torch
from metatomic.torch import ModelOutput, System

from metatrain.experimental.lorem import LOREM
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saved = torch.load(sys.argv[1], weights_only=False)
target = get_energy_target_info("energy", {"quantity": "energy", "unit": "eV"}, add_position_gradients=True)


def build(state):
    torch.manual_seed(42)
    m = LOREM(dict(saved["hypers"]), DatasetInfo(length_unit="angstrom", atomic_types=saved["atomic_types"], targets={"energy": target}))
    if state is not None:
        print("load:", m.load_state_dict(state, strict=False), flush=True)
    return m.to(dev)


def check(m, s):
    nl = m.requested_neighbor_lists()
    # fresh system: the saved ones carry the training run's neighbour lists
    s = System(types=s['types'], positions=s['positions'].to(torch.float32), cell=s['cell'].to(torch.float32), pbc=s['pbc'])
    s = get_system_with_neighbor_lists(s, nl).to(dev)
    s.positions.requires_grad_(True)
    m.zero_grad()
    e = m([s], {"energy": ModelOutput(sample_kind="system")})["energy"].block().values.sum()
    (g,) = torch.autograd.grad(e, s.positions, create_graph=True)
    (e**2 + (g**2).sum()).backward()
    bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    d = s.get_neighbor_list(nl[0]).values.squeeze(-1).norm(dim=-1)
    return dict(n_atoms=len(s), pbc=bool(s.pbc.all()), min_dist=round(float(d.min()), 4) if len(d) else None,
                zero_pairs=int((d < 1e-6).sum()), E_ok=bool(torch.isfinite(e)), F_ok=bool(torch.isfinite(g).all()),
                nan_grad_params=len(bad), first=bad[:3])


for label, state in (("saved", saved["state"]), ("fresh", None)):
    m = build(state)
    print(f"== {label} weights", flush=True)
    for idx, s in zip(saved["index"], saved["systems"]):
        r = check(m, s)
        if r["nan_grad_params"] or not (r["E_ok"] and r["F_ok"]) or (r["min_dist"] is not None and r["min_dist"] < 0.5):
            print(f"index {int(idx)}", r, flush=True)
    print(f"== {label} done", flush=True)
