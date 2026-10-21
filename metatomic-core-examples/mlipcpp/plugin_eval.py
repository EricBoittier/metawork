"""Evaluate xyz files with the mlipcpp metatomic plugin, through metatomic_core.

usage: plugin_eval.py <plugin.so> <model.gguf> <out.npz> <xyz>...
"""
import sys

import ase.io
import metatensor
import numpy as np

import metatomic as mta

plugin, gguf, out, *files = sys.argv[1:]
mta.load_plugin(plugin)
metatensor.register_external_data_wrapper("metatensor::SimpleDataArray", metatensor.ExternalCpuArray)
model = mta.load_model(gguf, {"dtype": "float64"})
caps = model.capabilities()
print(caps)

energy = mta.Quantity(name="energy", unit="eV", sample_kind="system", gradients=["positions", "strain"])
nc = mta.Quantity(name="non_conservative_force", unit="eV/Angstrom", sample_kind="atom")


def system(atoms):
    cell = np.array(atoms.cell[:], dtype=np.float64)
    cell[~atoms.pbc] = 0.0
    return mta.System(
        "angstrom",
        atoms.numbers.astype(np.int32),
        atoms.positions.astype(np.float64),
        cell,
        atoms.pbc.astype(bool),
    )


frames = [ase.io.read(f) for f in files]
systems = [system(a) for a in frames]
e, f = mta.execute_model(model, systems, [energy, nc], check_consistency=True)

block = e.block()
results = {}
offset = 0
for i, (name, atoms) in enumerate(zip(files, frames)):
    n = len(atoms)
    key = name.split("/")[-1].removesuffix(".xyz")
    results[f"{key}/energy"] = block.values[i, 0]
    results[f"{key}/forces"] = -block.gradient("positions").values[offset : offset + n, :, 0]
    virial = block.gradient("strain").values[i, :, :, 0]
    results[f"{key}/virial"] = virial
    results[f"{key}/nc_forces"] = f.block().values[offset : offset + n, :, 0]
    offset += n
np.savez(out, **results)
print("saved", out)
