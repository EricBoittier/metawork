"""Run in the metatrain venv. Warm-starts JaxParityBackbone/JaxParityLongRange
from a dumped lorem-jax checkpoint (see ``dump_reference.py``) and runs a
short Adam fine-tune on the archive's own training data, to see how close a
warm start + brief training gets to the paper's reported accuracy -- not
just the raw checkpoint-loading parity that ``compare.py`` checks.

This is a smoke test, not a reproduction: plain Adam with a simple linear
decay in place of the paper's LAMB + full-length schedule, and whatever
epoch budget you pass on the command line rather than ``settings.yaml``'s
``max_epochs`` (which is typically 2000-4000). See ``README.md`` for a
worked example and what it reached on cumulene.

Usage::

    .venv/bin/python train_smoke.py \\
        lorem-tmlr-archive/evals/cumulene/lorem/run/checkpoints/R2_E+F/model/model.yaml \\
        cumulene_reference.npz \\
        lorem-tmlr-archive/datasets/cumulene_train.xyz \\
        lorem-tmlr-archive/datasets/cumulene_test.xyz \\
        --epochs 200 --batch-size 32
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from metatomic.torch import NeighborListOptions, System
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from compare import load_model_yaml, build_model
from metatrain.experimental.lorem.modules.jax_parity_checkpoint import load_checkpoint

DEVICE = torch.device(
    os.environ.get("LOREM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
)


def predict_batch(backbone, long_range, atoms_list, baseline, want_forces=False):
    systems = []
    positions_list = []
    for atoms in atoms_list:
        positions = torch.tensor(
            atoms.get_positions(), dtype=torch.float64, device=DEVICE,
            requires_grad=want_forces,
        )
        positions_list.append(positions)
        system = System(
            types=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=DEVICE),
            positions=positions,
            cell=torch.tensor(atoms.get_cell().array, dtype=torch.float64, device=DEVICE),
            pbc=torch.tensor(atoms.get_pbc(), device=DEVICE),
        )
        systems.append(get_system_with_neighbor_lists(system, [backbone.neighbor_list_options]))

    nodes_scalar, distances, nodes_spherical, sr_energy = backbone(systems)
    lr_energy = long_range(systems, nodes_scalar, distances, nodes_spherical)
    per_atom = sr_energy + lr_energy
    baseline_flat = torch.cat(
        [
            torch.tensor(
                [baseline[int(z)] for z in atoms.get_atomic_numbers()],
                dtype=torch.float64, device=DEVICE,
            )
            for atoms in atoms_list
        ]
    )
    per_atom = per_atom + baseline_flat

    sizes = [len(atoms) for atoms in atoms_list]
    energies, idx = [], 0
    for n in sizes:
        energies.append(per_atom[idx : idx + n].sum())
        idx += n
    energies = torch.stack(energies)

    forces = None
    if want_forces:
        grads = torch.autograd.grad(energies.sum(), positions_list, create_graph=torch.is_grad_enabled())
        forces = [-g for g in grads]
    return energies, forces


def evaluate(backbone, long_range, atoms_list, baseline, label):
    backbone.eval()
    long_range.eval()
    e_pred, f_pred = predict_batch(backbone, long_range, atoms_list, baseline, want_forces=True)
    e_ref = np.array([a.get_potential_energy() for a in atoms_list])
    f_ref = np.concatenate([a.get_forces() for a in atoms_list], axis=0)
    f_pred_flat = np.concatenate([f.detach().cpu().numpy() for f in f_pred], axis=0)
    sizes = np.array([len(a) for a in atoms_list])

    e_pred_np = e_pred.detach().cpu().numpy()
    e_rmse = np.sqrt(np.mean(((e_pred_np - e_ref) / sizes) ** 2)) * 1000
    f_rmse = np.sqrt(np.mean((f_pred_flat - f_ref) ** 2)) * 1000
    print(f"[{label}] energy RMSE={e_rmse:.3f} meV/atom  forces RMSE={f_rmse:.3f} meV/A")
    return e_rmse, f_rmse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_yaml", type=Path)
    parser.add_argument("reference_npz", type=Path, help="from dump_reference.py (params only needed)")
    parser.add_argument("train_xyz", type=Path)
    parser.add_argument("test_xyz", type=Path)
    parser.add_argument("--n-train", type=int, default=None, help="default: all frames")
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start-lr", type=float, default=1e-3)
    parser.add_argument("--end-lr", type=float, default=1e-4)
    args = parser.parse_args()

    print(f"device: {DEVICE}")
    hypers, baseline = load_model_yaml(args.model_yaml)
    reference = np.load(args.reference_npz)
    flax_params = {
        k[len("param__"):].replace("__SLASH__", "/"): v
        for k, v in reference.items()
        if k.startswith("param__")
    }

    nlo = NeighborListOptions(cutoff=hypers["cutoff"], full_list=True, strict=True)
    atomic_types = sorted(int(z) for z in baseline)
    backbone, long_range = build_model(hypers, atomic_types, nlo)
    backbone.to(DEVICE)
    long_range.to(DEVICE)
    load_checkpoint(backbone, long_range, flax_params)

    train_index = f"0:{args.n_train}" if args.n_train else ":"
    test_index = f"0:{args.n_test}" if args.n_test else ":"
    train_atoms = read(args.train_xyz, index=train_index)
    test_atoms = read(args.test_xyz, index=test_index)
    print(f"{len(train_atoms)} train frames, {len(test_atoms)} test frames")

    evaluate(backbone, long_range, test_atoms, baseline, "warm-start / test")

    params = list(backbone.parameters()) + list(long_range.parameters())
    optimizer = torch.optim.Adam(params, lr=args.start_lr)
    n_batches = (len(train_atoms) + args.batch_size - 1) // args.batch_size

    t0 = time.time()
    for epoch in range(args.epochs):
        frac = epoch / max(1, args.epochs - 1)
        lr = args.start_lr + frac * (args.end_lr - args.start_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        backbone.train()
        long_range.train()
        perm = np.random.permutation(len(train_atoms))
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
            batch_atoms = [train_atoms[i] for i in idx]
            optimizer.zero_grad()
            e_pred, f_pred = predict_batch(backbone, long_range, batch_atoms, baseline, want_forces=True)
            e_ref = torch.tensor(
                [a.get_potential_energy() for a in batch_atoms], dtype=torch.float64, device=DEVICE
            )
            sizes = torch.tensor([len(a) for a in batch_atoms], dtype=torch.float64, device=DEVICE)
            e_loss = (((e_pred - e_ref) / sizes) ** 2).mean()
            f_loss = 0.0
            for f, a in zip(f_pred, batch_atoms):
                f_ref = torch.tensor(a.get_forces(), dtype=torch.float64, device=DEVICE)
                f_loss = f_loss + ((f - f_ref) ** 2).mean()
            f_loss = f_loss / len(batch_atoms)
            loss = 0.5 * e_loss + 0.5 * f_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"epoch {epoch}: lr={lr:.2e} loss={epoch_loss / n_batches:.5f} ({time.time() - t0:.0f}s)")

    evaluate(backbone, long_range, test_atoms, baseline, "fine-tuned / test")


if __name__ == "__main__":
    main()
