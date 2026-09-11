"""Run in the metatrain venv (metawork/.venv). No JAX/flax import needed.

Loads a ``dump_reference.py`` output into ``JaxParityBackbone`` /
``JaxParityLongRange`` (via ``jax_parity_checkpoint.load_checkpoint``) and
compares energies/forces against the JAX reference it shipped with.

Usage::

    .venv/bin/python compare.py \\
        lorem-tmlr-archive/evals/AuMgO/lorem/run/checkpoints/R2_E+F/model/model.yaml \\
        lorem-tmlr-archive/datasets/AuMgO_valid.xyz \\
        aumgo_reference.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from ase.io import read
from metatomic.torch import NeighborListOptions, System
from metatrain.experimental.lorem.modules.jax_parity import (
    JaxParityBackbone,
    JaxParityLongRange,
)
from metatrain.experimental.lorem.modules.jax_parity_checkpoint import load_checkpoint
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

# lorem.models.mlip.Lorem's own field defaults -- model.yaml only lists the
# overrides, so anything it doesn't mention falls back to these.
LOREM_DEFAULTS = dict(
    cutoff=5.0,
    max_degree=6,
    max_degree_lr=2,
    num_features=128,
    num_radial=32,
    num_species=8,
    num_spherical_features=8,
)


def load_model_yaml(path: Path):
    with open(path) as f:
        doc = yaml.safe_load(f)
    hypers = dict(LOREM_DEFAULTS)
    hypers.update(doc["model"]["lorem.Lorem"])
    baseline = doc["baseline"]["elemental"]
    return hypers, baseline


def build_model(hypers, atomic_types, nlo):
    backbone = JaxParityBackbone(
        cutoff=hypers["cutoff"],
        max_degree=hypers["max_degree"],
        num_features=hypers["num_features"],
        num_radial=hypers["num_radial"],
        num_spherical_features=hypers["num_spherical_features"],
        num_species=hypers["num_species"],
        atomic_types=atomic_types,
        neighbor_list_options=nlo,
    ).double()
    long_range = JaxParityLongRange(
        feature_dim=hypers["num_features"],
        num_spherical_features=hypers["num_spherical_features"],
        max_degree=hypers["max_degree"],
        max_degree_lr=hypers["max_degree_lr"],
        neighbor_list_options=nlo,
        # Not read from model.yaml -- lorem-jax's own marathon.prepare()
        # derives these from the model's cutoff at data-prep time:
        # smearing = cutoff/4, lr_wavelength = cutoff/8. Use that formula,
        # not a torchpme.tuning.ewald.tune_ewald guess (tune_ewald's own
        # recommendation reproduces the *converged* Ewald sum just as well,
        # but only marathon's specific choice matches what the checkpoint
        # was actually trained against bit-for-bit).
        smearing=hypers["cutoff"] / 4.0,
        kspace_resolution=hypers["cutoff"] / 8.0,
    ).double()
    return backbone, long_range


def predict(backbone, long_range, atoms, baseline, want_forces=False):
    positions = torch.tensor(
        atoms.get_positions(), dtype=torch.float64, requires_grad=want_forces
    )
    system = System(
        types=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32),
        positions=positions,
        cell=torch.tensor(atoms.get_cell().array, dtype=torch.float64),
        pbc=torch.tensor(atoms.get_pbc()),
    )
    nlo = backbone.neighbor_list_options
    system = get_system_with_neighbor_lists(system, [nlo])
    nodes_scalar, distances, nodes_spherical, sr_energy = backbone([system])
    lr_energy = long_range([system], nodes_scalar, distances, nodes_spherical)
    per_atom_baseline = torch.tensor(
        [baseline[int(z)] for z in atoms.get_atomic_numbers()], dtype=torch.float64
    )
    total_energy = (sr_energy + lr_energy + per_atom_baseline).sum()
    if not want_forces:
        return total_energy.item(), None
    (grad,) = torch.autograd.grad(total_energy, positions)
    return total_energy.item(), (-grad).detach().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_yaml", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("reference_npz", type=Path)
    args = parser.parse_args()

    hypers, baseline = load_model_yaml(args.model_yaml)
    reference = np.load(args.reference_npz)
    flax_params = {
        k[len("param__"):].replace("__SLASH__", "/"): v
        for k, v in reference.items()
        if k.startswith("param__")
    }
    jax_energies = reference["jax_energies"]
    jax_forces = reference["jax_forces"]
    jax_force_sizes = reference["jax_force_sizes"]
    n_frames = len(jax_energies)

    nlo = NeighborListOptions(cutoff=hypers["cutoff"], full_list=True, strict=True)
    atomic_types = sorted(int(z) for z in baseline)
    backbone, long_range = build_model(hypers, atomic_types, nlo)
    load_checkpoint(backbone, long_range, flax_params)
    backbone.eval()
    long_range.eval()

    atoms_list = read(args.dataset, index=f"0:{n_frames}")
    torch_energies = []
    force_start = 0
    force_diffs = []
    for atoms, n_atoms in zip(atoms_list, jax_force_sizes):
        energy, forces = predict(backbone, long_range, atoms, baseline, want_forces=True)
        torch_energies.append(energy)
        force_diffs.append(forces - jax_forces[force_start : force_start + n_atoms])
        force_start += n_atoms

    torch_energies = np.array(torch_energies)
    sizes = np.array([len(a) for a in atoms_list])
    e_diff_mev_atom = (torch_energies - jax_energies) / sizes * 1000
    f_diff = np.concatenate(force_diffs, axis=0) * 1000

    print(f"{n_frames} frames from {args.dataset.name}, checkpoint {args.model_yaml.parent.parent}")
    print(
        f"energy: RMSE={np.sqrt(np.mean(e_diff_mev_atom**2)):.3f} "
        f"MAE={np.mean(np.abs(e_diff_mev_atom)):.3f} meV/atom (torch port vs. JAX reference)"
    )
    print(
        f"forces: RMSE={np.sqrt(np.mean(f_diff**2)):.3f} "
        f"MAE={np.mean(np.abs(f_diff)):.3f} meV/Angstrom"
    )


if __name__ == "__main__":
    main()
