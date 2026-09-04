# Inspect QM7-X HDF5 and converted files

The original dataset is eight HDF5 files. Each file is a nested group:

```
<molecule id>/                # e.g. "1", "94", "100"
  <conformation id>/          # Geom-mr-is-ct-u  (see below)
    atNUM, atXYZ, eAT, …      # datasets on the conformation group
```

`idconf` has the form `Geom-mr-is-ct-u`:

| token | meaning |
| --- | --- |
| `r` | SMILES string index |
| `s` | stereoisomer (excluding conformers) |
| `t` | (meta)stable conformer |
| `u` | `opt` = DFTB3+MBD optimized, or `1`…`100` = displaced non-equilibrium |

Those indices are **not** sorted by PBE0+MBD energy.


## Scripts

Use [`run.sh`](run.sh) so the metawork venv (and `h5py`) is picked up even
if that venv is not activated:

```bash
bash etc/qm7x_zenodo/inspect/run.sh list_hdf5 ~/data/qm7x/raw/8000.hdf5
bash etc/qm7x_zenodo/inspect/run.sh example_read_conf ~/data/qm7x/raw/8000.hdf5 --tmap
bash etc/qm7x_zenodo/inspect/run.sh summarize_xyz ~/data/qm7x/qm7x.xyz
```

[`hdf5-properties.md`](hdf5-properties.md) is the full key list from the
Zenodo `README.txt`, with a column for whether this example maps it into
a metatrain target.


## Duplicates

Zenodo ships `DupMols.dat`: some equilibrium structures appear in more
than one shard. `convert.sh` copies it into `$DATA_DIR/raw/` next to the
HDF5. The authors' `createDB.py` can drop those molecules **and** their
100 displaced structures (uncomment their line 55). This example does
**not** drop them — a subsample of `8000.xz` is unlikely to hit the list,
and the displaced geometries are not identical. Decide before a full-dump
training run.


## Adding another HDF5 property

1. Confirm the key, shape, and unit in [`hdf5-properties.md`](hdf5-properties.md)
   or `list_hdf5.py`.
2. Extend `load_structure()` / `to_atoms()` / `to_targets()` in
   [`../zenodo_to_metatensor.py`](../zenodo_to_metatensor.py).
3. Copy an options YAML whose `sample_kind` / `type` match (scalar vs
   Cartesian vs spherical, system vs atom). `c6-pet.yaml` and
   `atomic-c6-pet.yaml` are the templates for optional keys.

Do not add a sparse key to a `--format zip` run unless every structure
has it — DiskDataset zips must be homogeneous.
