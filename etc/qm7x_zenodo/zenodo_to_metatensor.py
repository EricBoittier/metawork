#!/usr/bin/env python3
"""Download a QM7-X HDF5 shard from Zenodo and convert it to metatensor /
metatrain formats, keeping several endpoints of different shape.

The original QM7-X dump (https://zenodo.org/records/4288677) stores ~40
properties per structure. OpenQDC only wraps energies and forces; this
script reads the HDF5 groups directly.

Default shard is ``8000.xz`` (~89 MB), the file the QM7-X README
recommends for a first pass. Outputs:

* ``xyz`` — extended XYZ with ASE-readable scalars/vectors/rank-2 tensors,
  plus ``polarizability_spherical.mts`` (λ=0 + λ=2; ASE cannot store two
  spherical irreps)
* ``zip`` — ``DiskDataset`` with one folder per structure, every endpoint
  a ``.mts`` file (including the spherical polarizability)

Examples:

    python zenodo_to_metatensor.py --n-samples 200
    python zenodo_to_metatensor.py --hdf5 8000.hdf5 --format zip
    python zenodo_to_metatensor.py --file 1000.xz --n-samples 500
"""

from __future__ import annotations

import argparse
import lzma
import shutil
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import torch
from ase import Atoms
from ase.io import write
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import systems_to_torch

from metatrain.utils.data.writers import DiskDatasetWriter

ZENODO = "https://zenodo.org/records/4288677/files"
DEFAULT_SHARD = "8000.xz"


def _download(name: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    url = f"{ZENODO}/{name}?download=1"
    print(f"downloading {url} -> {dest}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "metawork-qm7x"})
    with urllib.request.urlopen(req) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)
    return dest


def _decompress_xz(xz_path: Path, hdf5_path: Path) -> Path:
    if hdf5_path.exists():
        return hdf5_path
    print(f"decompressing {xz_path} -> {hdf5_path}", flush=True)
    with lzma.open(xz_path) as src, open(hdf5_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return hdf5_path


def _as_array(value) -> np.ndarray:
    try:
        value = value[()]
    except Exception:
        pass
    return np.asarray(value)


def iter_confs(h5: h5py.File):
    for mol_id, mol in h5.items():
        if not isinstance(mol, h5py.Group):
            continue
        for conf_id, conf in mol.items():
            if isinstance(conf, h5py.Group) and "atNUM" in conf:
                yield str(mol_id), str(conf_id), conf


def _get(conf, key: str):
    if key not in conf:
        return None
    return _as_array(conf[key])


def l0_from_matrix(A: np.ndarray) -> float:
    """Isotropic (trace) part. Same convention as metatrain's test helper."""
    A = A.reshape(3, 3)
    return float(np.trace(A))


def l2_from_matrix(A: np.ndarray) -> np.ndarray:
    """λ=2 components of a symmetric 3×3 tensor, metatrain test convention."""
    A = A.reshape(3, 3)
    l2 = np.empty(5)
    l2[0] = (A[0, 1] + A[1, 0]) / 2.0
    l2[1] = (A[1, 2] + A[2, 1]) / 2.0
    l2[2] = (2.0 * A[2, 2] - A[0, 0] - A[1, 1]) / (2.0 * np.sqrt(3.0))
    l2[3] = (A[0, 2] + A[2, 0]) / 2.0
    l2[4] = (A[0, 0] - A[1, 1]) / 2.0
    return l2


def _labels_xyz() -> Labels:
    return Labels("xyz", torch.tensor([[0], [1], [2]]))


def scalar_system(value: float, system: int) -> TensorMap:
    return TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=torch.tensor([[value]], dtype=torch.float64),
                samples=Labels("system", torch.tensor([[system]])),
                components=[],
                properties=Labels.single(),
            )
        ],
    )


def scalar_atom(values: np.ndarray, system: int) -> TensorMap:
    n = len(values)
    return TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=torch.tensor(values, dtype=torch.float64).reshape(n, 1),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[system, i] for i in range(n)]),
                ),
                components=[],
                properties=Labels.single(),
            )
        ],
    )


def cartesian_system(vector: np.ndarray, system: int, rank: int = 1) -> TensorMap:
    tensor = np.asarray(vector, dtype=np.float64)
    if rank == 1:
        values = torch.tensor(tensor.reshape(1, 3, 1), dtype=torch.float64)
        components = [_labels_xyz()]
    elif rank == 2:
        values = torch.tensor(tensor.reshape(1, 3, 3, 1), dtype=torch.float64)
        components = [
            Labels("xyz_1", torch.tensor([[0], [1], [2]])),
            Labels("xyz_2", torch.tensor([[0], [1], [2]])),
        ]
    else:
        raise ValueError(rank)
    return TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=values,
                samples=Labels("system", torch.tensor([[system]])),
                components=components,
                properties=Labels.single(),
            )
        ],
    )


def cartesian_atom(array: np.ndarray, system: int) -> TensorMap:
    n = array.shape[0]
    return TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=torch.tensor(array, dtype=torch.float64).reshape(n, 3, 1),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[system, i] for i in range(n)]),
                ),
                components=[_labels_xyz()],
                properties=Labels.single(),
            )
        ],
    )


def energy_map(energy: float, forces: np.ndarray | None, system: int) -> TensorMap:
    block = TensorBlock(
        values=torch.tensor([[energy]], dtype=torch.float64),
        samples=Labels("system", torch.tensor([[system]])),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    if forces is not None:
        n = forces.shape[0]
        block.add_gradient(
            "positions",
            TensorBlock(
                values=-torch.tensor(forces, dtype=torch.float64).unsqueeze(-1),
                samples=Labels(
                    names=["sample", "atom"],
                    values=torch.tensor([[0, i] for i in range(n)]),
                ),
                components=[_labels_xyz()],
                properties=Labels("energy", torch.tensor([[0]])),
            ),
        )
    return TensorMap(Labels.single(), [block])


def polarizability_spherical(
    matrices: list[np.ndarray], system_ids: list[int]
) -> TensorMap:
    l0 = np.array([l0_from_matrix(m) for m in matrices], dtype=np.float64)
    l2 = np.array([l2_from_matrix(m) for m in matrices], dtype=np.float64)
    samples = Labels(
        "system", torch.tensor(system_ids, dtype=torch.int32).reshape(-1, 1)
    )
    properties = Labels.single()
    block0 = TensorBlock(
        values=torch.tensor(l0).reshape(len(matrices), 1, 1),
        samples=samples,
        components=[Labels.range("o3_mu", 1)],
        properties=properties,
    )
    block2 = TensorBlock(
        values=torch.tensor(l2).reshape(len(matrices), 5, 1),
        samples=samples,
        components=[
            Labels("o3_mu", torch.tensor([[-2], [-1], [0], [1], [2]])),
        ],
        properties=properties,
    )
    keys = Labels(
        names=["o3_lambda", "o3_sigma"],
        values=torch.tensor([[0, 1], [2, 1]]),
    )
    return TensorMap(keys, [block0, block2])


def load_structure(conf) -> dict | None:
    numbers = _get(conf, "atNUM")
    positions = _get(conf, "atXYZ")
    energy = _get(conf, "eAT")
    if numbers is None or positions is None or energy is None:
        return None
    numbers = numbers.astype(np.int32).reshape(-1)
    positions = positions.reshape(-1, 3)
    out = {
        "numbers": numbers,
        "positions": positions,
        "energy": float(np.asarray(energy).reshape(-1)[0]),
        "forces": None,
        "hlgap": None,
        "dipole": None,
        "hirshfeld_charge": None,
        "hirshfeld_dipole": None,
        "polarizability": None,
        "c6": None,
        "atomic_c6": None,
        "name": None,
    }
    forces = _get(conf, "totFOR")
    if forces is not None:
        out["forces"] = forces.reshape(-1, 3)
    gap = _get(conf, "HLgap")
    if gap is not None:
        out["hlgap"] = float(np.asarray(gap).reshape(-1)[0])
    dip = _get(conf, "vDIP")
    if dip is not None:
        out["dipole"] = dip.reshape(3)
    hchg = _get(conf, "hCHG")
    if hchg is not None:
        out["hirshfeld_charge"] = hchg.reshape(-1)
    hdip = _get(conf, "hVDIP")
    if hdip is not None:
        out["hirshfeld_dipole"] = hdip.reshape(-1, 3)
    pol = _get(conf, "mTPOL")
    if pol is not None:
        out["polarizability"] = pol.reshape(3, 3)
    c6 = _get(conf, "mC6")
    if c6 is not None:
        out["c6"] = float(np.asarray(c6).reshape(-1)[0])
    atc6 = _get(conf, "atC6")
    if atc6 is not None:
        out["atomic_c6"] = atc6.reshape(-1)
    return out


def to_atoms(entry: dict) -> Atoms:
    atoms = Atoms(numbers=entry["numbers"], positions=entry["positions"])
    atoms.info["energy"] = entry["energy"]
    atoms.info["name"] = entry.get("name") or ""
    if entry["forces"] is not None:
        atoms.arrays["forces"] = entry["forces"]
    if entry["hlgap"] is not None:
        atoms.info["hlgap"] = entry["hlgap"]
    if entry["dipole"] is not None:
        atoms.info["dipole"] = entry["dipole"]
    if entry["hirshfeld_charge"] is not None:
        atoms.arrays["hirshfeld_charge"] = entry["hirshfeld_charge"]
    if entry["hirshfeld_dipole"] is not None:
        atoms.arrays["hirshfeld_dipole"] = entry["hirshfeld_dipole"]
    if entry["polarizability"] is not None:
        atoms.info["polarizability"] = entry["polarizability"].reshape(9)
    if entry["c6"] is not None:
        atoms.info["c6"] = entry["c6"]
    if entry["atomic_c6"] is not None:
        atoms.arrays["atomic_c6"] = entry["atomic_c6"]
    return atoms


def to_targets(entry: dict, system: int) -> dict[str, TensorMap]:
    targets = {"energy": energy_map(entry["energy"], entry["forces"], system)}
    if entry["hlgap"] is not None:
        targets["hlgap"] = scalar_system(entry["hlgap"], system)
    if entry["dipole"] is not None:
        targets["dipole"] = cartesian_system(entry["dipole"], system, rank=1)
    if entry["hirshfeld_charge"] is not None:
        targets["hirshfeld_charge"] = scalar_atom(entry["hirshfeld_charge"], system)
    if entry["hirshfeld_dipole"] is not None:
        targets["hirshfeld_dipole"] = cartesian_atom(entry["hirshfeld_dipole"], system)
    if entry["polarizability"] is not None:
        targets["polarizability"] = polarizability_spherical(
            [entry["polarizability"]], [system]
        )
    if entry["c6"] is not None:
        targets["c6"] = scalar_system(entry["c6"], system)
    if entry["atomic_c6"] is not None:
        targets["atomic_c6"] = scalar_atom(entry["atomic_c6"], system)
    return targets


REQUIRED_HDF5 = (
    "atNUM",
    "atXYZ",
    "eAT",
    "totFOR",
    "HLgap",
    "vDIP",
    "hCHG",
    "hVDIP",
    "mTPOL",
)


def collect_entries(hdf5_path: Path, n_samples: str, seed: int) -> list[dict]:
    with h5py.File(hdf5_path, "r") as h5:
        keys = [
            (mol, conf)
            for mol, conf, group in iter_confs(h5)
            if all(k in group for k in REQUIRED_HDF5)
        ]
        print(
            f"{hdf5_path}: {len(keys)} structures with the full endpoint set",
            flush=True,
        )
        if n_samples != "all":
            n = min(int(n_samples), len(keys))
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(keys), size=n, replace=False))
            keys = [keys[i] for i in idx]
        entries = []
        for mol_id, conf_id in keys:
            entry = load_structure(h5[mol_id][conf_id])
            if entry is None:
                continue
            entry["name"] = f"{mol_id}/{conf_id}"
            entries.append(entry)
    return entries


def write_xyz(entries: list[dict], xyz_path: Path, mts_path: Path) -> None:
    frames = [to_atoms(e) for e in entries]
    write(xyz_path, frames)
    matrices = [e["polarizability"] for e in entries]
    polarizability_spherical(matrices, list(range(len(entries)))).save(str(mts_path))


def write_zip(entries: list[dict], zip_path: Path) -> None:
    writer = DiskDatasetWriter(zip_path)
    for i, entry in enumerate(entries):
        atoms = to_atoms(entry)
        system = systems_to_torch(atoms, dtype=torch.float64)
        writer.write([system], to_targets(entry, system=0))
        if (i + 1) % 50 == 0:
            print(f"  wrote {i + 1}/{len(entries)}", flush=True)
    writer.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        default=DEFAULT_SHARD,
        help="Zenodo filename to fetch (default: 8000.xz)",
    )
    parser.add_argument(
        "--hdf5",
        default=None,
        help="use an already-decompressed HDF5 file (skips download)",
    )
    parser.add_argument("--cache-dir", default="qm7x_raw")
    parser.add_argument("--n-samples", default="200")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=("xyz", "zip"), default="xyz")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    if args.hdf5:
        hdf5_path = Path(args.hdf5)
    else:
        xz_path = _download(args.file, cache / args.file)
        hdf5_name = args.file.removesuffix(".xz") + ".hdf5"
        hdf5_path = _decompress_xz(xz_path, cache / hdf5_name)

    entries = collect_entries(hdf5_path, args.n_samples, args.seed)
    if not entries:
        raise SystemExit(f"no usable structures in {hdf5_path}")

    if args.format == "xyz":
        xyz = Path(args.output or "qm7x.xyz")
        mts = xyz.with_name("polarizability_spherical.mts")
        write_xyz(entries, xyz, mts)
        print(f"wrote {len(entries)} structures to {xyz} and {mts}")
    else:
        zpath = Path(args.output or "qm7x.zip")
        write_zip(entries, zpath)
        print(f"wrote {len(entries)} structures to {zpath}")


if __name__ == "__main__":
    main()
