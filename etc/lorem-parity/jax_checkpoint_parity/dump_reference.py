"""Run in the lorem-jax venv (e.g. metawork/.venv-lorem-jax).

Dumps a shipped lorem-jax checkpoint's flax parameter tree to a flat
``.npz`` (no JAX needed to read it back), and computes JAX reference
energies/forces on a handful of frames from a dataset, for
``compare.py`` (run in the metatrain venv) to check against.

Usage::

    .venv-lorem-jax/bin/python dump_reference.py \\
        lorem-tmlr-archive/evals/AuMgO/lorem/run/checkpoints/R2_E+F \\
        lorem-tmlr-archive/datasets/AuMgO_valid.xyz \\
        --n-frames 20 --out aumgo_reference.npz
"""

import argparse
from pathlib import Path

import numpy as np
from ase.io import read
from lorem.calculator import Calculator
from marathon.emit.checkpoint import read_msgpack


def flatten_flax_tree(tree: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_flax_tree(value, name))
        else:
            flat[name] = np.asarray(value)
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="checkpoint dir, e.g. .../run/checkpoints/R2_E+F")
    parser.add_argument("dataset", type=Path, help="an extended-XYZ file with energy/forces")
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("reference.npz"))
    args = parser.parse_args()

    params = read_msgpack(args.checkpoint / "model/model.msgpack")
    flat_params = flatten_flax_tree(params["params"])

    calc = Calculator.from_checkpoint(args.checkpoint)
    atoms_list = read(args.dataset, index=f"0:{args.n_frames}")

    energies = []
    forces = []
    for atoms in atoms_list:
        atoms.calc = calc
        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())

    np.savez(
        args.out,
        **{f"param__{k.replace('/', '__SLASH__')}": v for k, v in flat_params.items()},
        jax_energies=np.array(energies),
        jax_forces=np.concatenate(forces, axis=0),
        jax_force_sizes=np.array([len(a) for a in atoms_list]),
    )
    print(f"wrote {args.out}: {len(flat_params)} checkpoint leaves, {len(atoms_list)} reference frames")


if __name__ == "__main__":
    main()
