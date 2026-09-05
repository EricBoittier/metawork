#!/usr/bin/env python3
"""Verbose architecture-key dump and DFT comparison numbers.

Run from the metawork root with the PyTorch / mtt env (not the JAX venv):

    metatrain/.tox/lorem-tests/bin/python etc/lorem-parity/compare.py

This prints the lorem-jax ``Lorem`` field names next to experimental.lorem
hypers, then scores a trained ``model.pt`` against the DFT labels on the
shared toy XYZ. It does **not** import JAX and does **not** claim the two
stacks predict the same energy.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
XYZ = ROOT / "lorem-jax/examples/train-mlp/data.xyz"
JAX_MLIP = ROOT / "lorem-jax/src/lorem/models/mlip.py"
DEFAULT_MODEL = ROOT / "outputs/2026-09-05/10-21-57/model.pt"


# Fallback if the submodule cannot be parsed (same table as the CI test).
LOREM_JAX_LOREM_DEFAULTS = {
    "cutoff": 5.0,
    "max_degree": 6,
    "max_degree_lr": 2,
    "num_features": 128,
    "num_radial": 32,
    "num_species": 8,
    "num_spherical_features": 8,
    "cutoff_fn": "cosine_cutoff",
    "radial_basis": "basic_bernstein",
    "lr": True,
    "num_message_passing": 0,
    "equivariant_message_passing": True,
    "initialize_node_features": True,
}

SHARED_HYPER_KEYS = (
    "cutoff",
    "max_degree",
    "max_degree_lr",
    "num_features",
    "num_radial",
    "num_spherical_features",
    "radial_basis",
    "num_message_passing",
)

IRIS_PETLR_KEYS = (
    "d_pet",
    "d_node",
    "d_head",
    "d_feedforward",
    "num_heads",
    "num_attention_layers",
    "num_gnn_layers",
    "cutoff",
    "cutoff_width",
    "lr",
    "num_charges",
    "lr_scale_init",
)


def _parse_lorem_jax_defaults(path: Path) -> dict:
    """Read ``class Lorem`` assignments from source. No JAX import."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Lorem":
            defaults = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    if stmt.value is None:
                        continue
                    defaults[stmt.target.id] = ast.literal_eval(stmt.value)
            return defaults
    raise ValueError(f"class Lorem not found in {path}")


def _print_table(title: str, rows: list[tuple[str, str]]) -> None:
    print()
    print(title)
    print("=" * len(title))
    if not rows:
        print("  (empty)")
        return
    width = max(len(row[0]) for row in rows)
    for left, right in rows:
        print(f"  {left:<{width}}  {right}")


def _nested_keys(tree: dict, prefix: str = "") -> list[str]:
    keys: list[str] = []
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            keys.extend(_nested_keys(value, name))
        else:
            keys.append(name)
    return keys


def dump_architecture_keys() -> dict:
    """Print both param dictionaries and the overlap."""
    from metatrain.utils.architectures import get_default_hypers

    jax_defaults = dict(LOREM_JAX_LOREM_DEFAULTS)
    source = "hardcoded fallback"
    if JAX_MLIP.is_file():
        jax_defaults = _parse_lorem_jax_defaults(JAX_MLIP)
        source = str(JAX_MLIP.relative_to(ROOT))

    hypers = get_default_hypers("experimental.lorem")["model"]

    print()
    print("Architecture param dictionaries")
    print("-------------------------------")
    print(
        "experimental.lorem is a metatrain-native port of lorem.Lorem. "
        "Shared knobs should match. Extra torch keys are metatrain/iris "
        "packaging. Extra jax keys are inferred or renamed, not missing physics."
    )
    print(f"lorem-jax defaults parsed from: {source}")

    _print_table(
        "lorem-jax Lorem field names (the JAX params / dataclass keys)",
        [(key, repr(value)) for key, value in jax_defaults.items()],
    )
    _print_table(
        "experimental.lorem model hypers keys (the torch params dict)",
        [
            (
                key,
                repr(hypers[key]) if not isinstance(hypers[key], dict) else "{...}",
            )
            for key in hypers
        ],
    )
    _print_table(
        "experimental.lorem nested keys",
        [(key, "") for key in _nested_keys(hypers)],
    )

    matches = 0
    rows = []
    for key in SHARED_HYPER_KEYS:
        jax_value = jax_defaults[key]
        torch_value = hypers[key]
        ok = jax_value == torch_value
        matches += int(ok)
        rows.append(
            (
                key,
                f"jax={jax_value!r}  torch={torch_value!r}  "
                f"[{'match' if ok else 'DIFF'}]",
            )
        )
    lr_ok = jax_defaults["lr"] == hypers["long_range"]["enable"]
    matches += int(lr_ok)
    rows.append(
        (
            "lr → long_range.enable",
            f"jax={jax_defaults['lr']!r}  "
            f"torch={hypers['long_range']['enable']!r}  "
            f"[{'match' if lr_ok else 'DIFF'}]",
        )
    )
    _print_table(
        f"shared knobs ({matches}/{len(SHARED_HYPER_KEYS) + 1} default values match)",
        rows,
    )

    jax_only = [
        key
        for key in jax_defaults
        if key not in SHARED_HYPER_KEYS and key != "lr"
    ]
    _print_table(
        "lorem-jax-only keys (covered under another name)",
        [
            (
                "num_species",
                "torch Embedding(max_Z+1, num_features); dataset types",
            ),
            (
                "cutoff_fn",
                "torch cosine cutoff via cutoff_width (paper 0.5 Å)",
            ),
            (
                "equivariant_message_passing",
                "on whenever num_message_passing > 0 (TensorDense path)",
            ),
            (
                "initialize_node_features",
                "always on: sr.species_embedding",
            ),
        ]
        if set(jax_only) <= {
            "num_species",
            "cutoff_fn",
            "equivariant_message_passing",
            "initialize_node_features",
        }
        else [(key, "see README") for key in jax_only],
    )
    _print_table(
        "experimental.lorem-only keys (metatrain / iris packaging)",
        [
            ("cutoff_width", "cosine envelope width; jax uses cutoff_fn"),
            ("sh_convention", "e3x Racah vs orthonormal sphericart"),
            ("trunk", "spherical (paper) or pet (iris-style)"),
            ("pet", "PETBackend overrides when trunk=pet"),
            ("long_range.use_ewald", "Ewald vs P3M during training"),
            ("long_range.lr_scale_init", "iris PETLR residual gate"),
        ],
    )
    _print_table(
        "iris PETLR keys (different trunk: PET + scalar charges)",
        [(key, "not a LOREM knob") for key in IRIS_PETLR_KEYS],
    )
    return {"jax": jax_defaults, "torch": hypers, "shared_matches": matches}


def dump_torch_param_keys() -> None:
    """Print the trained-model-shaped module tree (small hypers, not paper 128)."""
    import copy

    from metatrain.experimental.lorem import LOREM
    from metatrain.utils.architectures import get_default_hypers
    from metatrain.utils.data import DatasetInfo
    from metatrain.utils.data.target_info import get_energy_target_info

    hypers = copy.deepcopy(get_default_hypers("experimental.lorem")["model"])
    hypers["max_degree"] = 1
    hypers["max_degree_lr"] = 0
    hypers["num_features"] = 16
    hypers["num_spherical_features"] = 2
    hypers["num_radial"] = 4
    hypers["long_range"]["enable"] = True
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    model = LOREM(hypers, dataset_info)
    names = list(model.named_parameters())
    n_params = sum(int(p.numel()) for _, p in names)
    prefixes: dict[str, int] = {}
    for name, param in names:
        prefixes[name.split(".")[0]] = prefixes.get(name.split(".")[0], 0) + int(
            param.numel()
        )

    _print_table(
        f"experimental.lorem parameter-tree keys ({n_params} scalars, small hypers)",
        [(name, str(tuple(param.shape))) for name, param in names],
    )
    _print_table(
        "top-level scopes (iris-style sr / lr)",
        [(prefix, f"{count} scalars") for prefix, count in sorted(prefixes.items())],
    )
    _print_table(
        "how those scopes map onto lorem-jax modules",
        [
            ("sr.species_embedding", "Initial / ChemicalEmbedding"),
            ("sr.feature_mlp / sr.mp_layers", "RadialCoefficients + Update"),
            ("sr.tensor_dense", "e3x.nn.TensorDense self-product"),
            ("sr.norm_update", "degree-norm mix, (2ℓ+1)^{1/4}"),
            ("lr.scalar_charge_mlp", "MLP([2*d, 1]) scalar charges"),
            ("lr.spherical_charge_dense", "TensorDense → max_degree_lr"),
            ("lr.lr_scale", "iris PETLR residual gate (not in lorem-jax)"),
            ("readouts.energy", "final MLP energy head"),
        ],
    )


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def compare_to_dft(model_path: Path) -> dict:
    """Score an exported experimental.lorem model against the toy XYZ labels."""
    from ase.io import read
    from metatomic_ase import MetatomicCalculator

    frames = read(str(XYZ), index=":", format="extxyz")
    ref_e = np.array([float(atoms.calc.results["energy"]) for atoms in frames])
    ref_f = [np.asarray(atoms.calc.results["forces"], dtype=float) for atoms in frames]
    n_atoms = np.array([len(atoms) for atoms in frames], dtype=float)

    calc = MetatomicCalculator(str(model_path), device="cpu")
    pred_e = []
    pred_f = []
    for atoms in frames:
        atoms = atoms.copy()
        atoms.calc = calc
        pred_e.append(float(atoms.get_potential_energy()))
        pred_f.append(np.asarray(atoms.get_forces(), dtype=float))
    pred_e = np.asarray(pred_e)

    energy_rmse = _rmse(pred_e, ref_e)
    energy_mae = _mae(pred_e, ref_e)
    energy_rmse_pa = _rmse(pred_e / n_atoms, ref_e / n_atoms)
    energy_mae_pa = _mae(pred_e / n_atoms, ref_e / n_atoms)
    force_rmse = _rmse(np.concatenate(pred_f), np.concatenate(ref_f))
    force_mae = _mae(np.concatenate(pred_f), np.concatenate(ref_f))
    corr = float(np.corrcoef(pred_e, ref_e)[0, 1])

    print()
    print("Comparison numbers vs DFT labels")
    print("================================")
    print(f"data:   {XYZ.relative_to(ROOT)}  ({len(frames)} frames, C/H, non-periodic)")
    print(f"model:  {model_path}")
    print("stack:  experimental.lorem (2-epoch toy train, small hypers)")
    print("claim:  fit-to-labels on this XYZ, not a JAX energy match")
    print()
    print(f"  energy RMSE          {energy_rmse:10.4f} eV")
    print(f"  energy MAE           {energy_mae:10.4f} eV")
    print(f"  energy RMSE / atom   {1000 * energy_rmse_pa:10.3f} meV/atom")
    print(f"  energy MAE  / atom   {1000 * energy_mae_pa:10.3f} meV/atom")
    print(f"  energy Pearson r     {corr:10.4f}")
    print(f"  force RMSE           {1000 * force_rmse:10.2f} meV/Å")
    print(f"  force MAE            {1000 * force_mae:10.2f} meV/Å")
    print()
    print("  frame   N    E_dft / eV    E_mtt / eV     Δ / meV")
    for i, (e_ref, e_pred, n) in enumerate(zip(ref_e, pred_e, n_atoms, strict=True)):
        print(
            f"  {i:5d}  {int(n):2d}  {e_ref:12.6f}  {e_pred:12.6f}  "
            f"{1000 * (e_pred - e_ref):8.2f}"
        )

    print()
    print("JAX side")
    print("--------")
    print(
        "No isolated JAX venv on this machine right now (disk was full "
        "while installing jaxlib). When one exists, train the official "
        "example and compare its RMSE-to-DFT on the same 21 frames:"
    )
    print("  cd lorem-jax/examples/train-mlp && DATASETS=. python prepare.py")
    print("  cd my_experiment && DATASETS=.. lorem-train")
    print(
        "Do not subtract E_mtt − E_jax and call that a bug: different SH, "
        "radial basis implementation, PME, and independent 2-epoch RNG."
    )
    return {
        "n_frames": len(frames),
        "energy_rmse_eV": energy_rmse,
        "energy_mae_eV": energy_mae,
        "energy_rmse_meV_per_atom": 1000 * energy_rmse_pa,
        "energy_mae_meV_per_atom": 1000 * energy_mae_pa,
        "energy_pearson_r": corr,
        "force_rmse_meV_A": 1000 * force_rmse,
        "force_mae_meV_A": 1000 * force_mae,
    }


def main() -> int:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    dump_architecture_keys()
    dump_torch_param_keys()
    if not model_path.is_file():
        print(f"\nNo exported model at {model_path}; skip DFT numbers.")
        print("Train first: mtt train etc/lorem-parity/options-jax-toy.yaml")
        return 0
    compare_to_dft(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
