"""Rewrite a MAD-CORE DiskDataset zip for conservative-force training.

``madcore-to-diskdataset --format zip`` stores forces as a separate
``non_conservative_force.mts`` target. Architectures that predict forces as
``-dE/dx`` (LOREM, PET with ``forces: true``) read them as the ``positions``
gradient of ``energy.mts`` instead, so this folds ``-F`` into that gradient
and copies ``system.mta`` and ``metadata/atom_counts.npy`` unchanged.

    python conservative_zip.py ~/data/madcore/mts-2pow18 ~/data/madcore/mts-2pow18-conservative
"""

import argparse
import io
import zipfile
from multiprocessing import Pool
from pathlib import Path

import metatensor
import numpy as np


def energy_with_forces(energy_bytes: bytes, force_bytes: bytes) -> bytes:
    load = lambda b: metatensor.io.load_buffer(np.load(io.BytesIO(b)))
    energy, force = load(energy_bytes).block(), load(force_bytes).block()
    atoms = force.samples.values
    block = metatensor.TensorBlock(
        energy.values, energy.samples, energy.components, energy.properties
    )
    block.add_gradient(
        "positions",
        metatensor.TensorBlock(
            -force.values,
            metatensor.Labels(
                ["sample", "atom"],
                np.column_stack([np.zeros(len(atoms), dtype=np.int32), atoms[:, 1]]),
            ),
            [metatensor.Labels(["xyz"], np.arange(3, dtype=np.int32)[:, None])],
            energy.properties,
        ),
    )
    keys = metatensor.Labels(["_"], np.array([[0]], dtype=np.int32))
    out = io.BytesIO()
    np.save(out, np.frombuffer(metatensor.io.save_buffer(metatensor.TensorMap(keys, [block])), dtype=np.uint8))
    return out.getvalue()


def entry(args):
    index, energy_bytes, force_bytes = args
    return index, energy_with_forces(energy_bytes, force_bytes)


def convert(source: Path, target: Path, workers: int) -> None:
    with zipfile.ZipFile(source) as src:
        names = set(src.namelist())
        n = sum(name.endswith("/system.mta") for name in names)
        read = lambda i, f: src.read(f"{i}/{f}")
        jobs = ((i, read(i, "energy.mts"), read(i, "non_conservative_force.mts")) for i in range(n))
        partial = target.with_name(target.name + ".part")
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as out, Pool(workers) as pool:
            for i, energy in pool.imap(entry, jobs, chunksize=256):
                out.writestr(f"{i}/system.mta", read(i, "system.mta"))
                out.writestr(f"{i}/energy.mts", energy)
            if "metadata/atom_counts.npy" in names:
                out.writestr("metadata/atom_counts.npy", src.read("metadata/atom_counts.npy"))
        partial.rename(target)
    print(f"{target}: {n} structures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="directory with train/val/test.zip")
    parser.add_argument("target", type=Path)
    parser.add_argument("-j", "--workers", type=int, default=16)
    args = parser.parse_args()
    args.target.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test", "train"):
        if not (args.target / f"{split}.zip").exists():
            convert(args.source / f"{split}.zip", args.target / f"{split}.zip", args.workers)
