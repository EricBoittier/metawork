"""Find the structures that make LOREM produce non-finite energies or forces.

For every structure of a DiskDataset zip: target finiteness, cell volume,
smallest interatomic distance and atoms without a neighbour inside the cutoff
(from the model's own neighbour list), then energy and forces (-dE/dx) of a
freshly initialised LOREM, short-range only and with the long-range block,
seeded like the training runs. Structures go through in batches; a non-finite
batch is re-run one structure at a time to name the culprits.

    python nan_scan.py DATA/mixed/train.zip OUT.npz [--limit N] [--batch 32]
"""

import argparse
import math

import numpy as np
import torch
from metatomic.torch import ModelOutput

from metatrain.experimental.lorem import LOREM
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo, DiskDataset
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(long_range, types):
    torch.manual_seed(42)
    hypers = get_default_hypers("experimental.lorem")["model"]
    hypers["long_range"] = long_range
    info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=types,
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}, add_position_gradients=True
            )
        },
    )
    return LOREM(hypers, info).eval().to(DEVICE)


def energy_forces(model, systems):
    for s in systems:
        s.positions.requires_grad_(True)
    out = model(systems, {"energy": ModelOutput(sample_kind="system")})
    e = out["energy"].block().values.squeeze(-1)
    grads = torch.autograd.grad(e.sum(), [s.positions for s in systems])
    for s in systems:
        s.positions.requires_grad_(False)
    return e.detach(), [-g for g in grads]


def finite(e, forces):
    return torch.isfinite(e).tolist(), [bool(torch.isfinite(f).all()) for f in forces]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("zip")
    p.add_argument("out")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()

    ds = DiskDataset(args.zip)
    n = min(len(ds), args.limit or len(ds))
    types = list(range(1, 103))
    models = {"sr": build(False, types), "lr": build(True, types)}
    nl = models["lr"].requested_neighbor_lists()
    cutoff = nl[0].cutoff
    print(f"device {DEVICE}; {n} structures, cutoff {cutoff} A, neighbour lists {len(nl)}", flush=True)

    rec = {k: np.zeros(n, dtype=dt) for k, dt in [
        ("n_atoms", int), ("periodic", bool), ("volume", float), ("min_dist", float),
        ("n_isolated", int), ("target_finite", bool), ("energy_per_atom", float),
        ("sr_e_ok", bool), ("sr_f_ok", bool), ("lr_e_ok", bool), ("lr_f_ok", bool)]}

    for start in range(0, n, args.batch):
        idx = list(range(start, min(start + args.batch, n)))
        systems = []
        for i in idx:
            sample = ds[i]
            s = get_system_with_neighbor_lists(sample.system.to(torch.float32), nl)
            block = sample.energy.block()
            ok = bool(torch.isfinite(block.values).all())
            if "positions" in block.gradients_list():
                ok &= bool(torch.isfinite(block.gradient("positions").values).all())
            pairs = s.get_neighbor_list(nl[0])
            centers = pairs.samples.column("first_atom")
            dist = pairs.values.squeeze(-1).norm(dim=-1)
            counts = torch.bincount(centers, minlength=len(s))
            rec["n_atoms"][i] = len(s)
            rec["periodic"][i] = bool(s.pbc.all())
            rec["volume"][i] = float(torch.det(s.cell.double()).abs()) if s.pbc.any() else math.nan
            rec["min_dist"][i] = float(dist.min()) if len(dist) else math.inf
            rec["n_isolated"][i] = int((counts == 0).sum())
            rec["target_finite"][i] = ok
            rec["energy_per_atom"][i] = float(block.values.sum()) / len(s)
            systems.append(s.to(DEVICE))

        for name, model in models.items():
            try:
                e_ok, f_ok = finite(*energy_forces(model, systems))
            except Exception as err:  # a batch that raises is re-run one by one too
                print(f"[{name}] batch {start}: {type(err).__name__}: {err}", flush=True)
                e_ok, f_ok = [False] * len(idx), [False] * len(idx)
            if not (all(e_ok) and all(f_ok)):
                e_ok, f_ok = [], []
                for s in systems:
                    try:
                        (a,), (b,) = finite(*energy_forces(model, [s]))
                    except Exception as err:
                        print(f"[{name}] single: {type(err).__name__}: {err}", flush=True)
                        a = b = False
                    e_ok.append(a)
                    f_ok.append(b)
            rec[f"{name}_e_ok"][idx] = e_ok
            rec[f"{name}_f_ok"][idx] = f_ok
        if start % (args.batch * 200) == 0:
            bad = {k: int((~rec[k][: idx[-1] + 1]).sum()) for k in ("target_finite", "sr_e_ok", "sr_f_ok", "lr_e_ok", "lr_f_ok")}
            print(f"{idx[-1] + 1}/{n} non-finite so far: {bad}", flush=True)

    np.savez(args.out, **rec)
    print("\n== summary")
    for k in ("target_finite", "sr_e_ok", "sr_f_ok", "lr_e_ok", "lr_f_ok"):
        bad = np.flatnonzero(~rec[k])
        print(f"{k}: {len(bad)} bad, first {bad[:10].tolist()}")
    print(f"isolated atoms: {int((rec['n_isolated'] > 0).sum())} structures; "
          f"min_dist < 0.5 A: {int((rec['min_dist'] < 0.5).sum())}")
    bad = np.flatnonzero(~(rec["sr_e_ok"] & rec["sr_f_ok"]) | ~(rec["lr_e_ok"] & rec["lr_f_ok"]))
    for i in bad[:30]:
        print(i, {k: rec[k][i].item() for k in rec})


if __name__ == "__main__":
    main()
