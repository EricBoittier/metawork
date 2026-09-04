# Convert QM7-X HDF5 → metatrain inputs

These scripts wrap [`../zenodo_to_metatensor.py`](../zenodo_to_metatensor.py).
They download a shard from
[Zenodo record 4288677](https://zenodo.org/records/4288677) on first run,
decompress it, and write the files the YAMLs in [`../options/`](../options/)
read (`qm7x.xyz` plus `polarizability_spherical.mts`, or `qm7x.zip`).

Default output directory is `~/data/qm7x` (override with `--data-dir` or
`QM7X_DATA_DIR`). Raw `.xz` / `.hdf5` files are cached under `raw/` so
re-runs do not download again. Convert also fetches the Zenodo `README.txt`
and `DupMols.dat` into `raw/` (small; skipped if already present).

`h5py` is installed by `etc/setup-metawork.sh`. If an older venv is missing
it: `uv pip install --python .venv/bin/python h5py`.


## Which script

| script | what it does |
| --- | --- |
| [`convert.sh`](convert.sh) | one shard, writes `$DATA_DIR/qm7x.xyz` (what `mtt train` expects) |
| [`convert_shard.sh`](convert_shard.sh) | one shard into `$DATA_DIR/shards/<id>/` |
| [`convert_all_shards.sh`](convert_all_shards.sh) | all eight shards into `shards/<id>/` |

```bash
# 200 random structures from the small 8000.xz shard — enough to train
bash etc/qm7x_zenodo/convert/convert.sh

# every conformation in that shard
bash etc/qm7x_zenodo/convert/convert.sh --n-samples all

# DiskDataset zip (for options/multi-zip-pet.yaml)
bash etc/qm7x_zenodo/convert/convert.sh --n-samples 200 --format zip

# a different shard, isolated so you can train on it alone
bash etc/qm7x_zenodo/convert/convert_shard.sh 1000 --n-samples 200

# subsample of every shard (still not the full 4.2 M)
bash etc/qm7x_zenodo/convert/convert_all_shards.sh

# the actual full dump — ~9.6 GB download, tens of GB uncompressed
bash etc/qm7x_zenodo/convert/convert_all_shards.sh --n-samples all
```

`--n-samples all` means **all structures in the chosen shard(s)**, not a
hidden extra dataset. The full QM7-X set is the eight shards together.


## Shards

Molecule IDs are partitioned across files (the number is the upper id,
not a count). `8000.xz` is the file the original README recommends for a
first pass.

| file | compressed | typical use |
| --- | --- | --- |
| `1000.xz` | ~682 MB | molecule ids up to 1000 |
| `2000.xz` | ~995 MB | |
| `3000.xz` | ~1.9 GB | |
| `4000.xz` | ~1.4 GB | |
| `5000.xz` | ~1.1 GB | |
| `6000.xz` | ~1.9 GB | |
| `7000.xz` | ~1.0 GB | |
| `8000.xz` | ~85 MB | default demo shard |

There is no `9000.xz`. Also on the record: `README.txt`, `createDB.py`
(their SchNetPack helper), and `DupMols.dat` (duplicated equilibrium
structures — see [`../inspect/hdf5-properties.md`](../inspect/hdf5-properties.md)).


## Layout after a demo convert

```
~/data/qm7x/
  qm7x.xyz
  polarizability_spherical.mts
  qm7x.zip                 # only with --format zip
  raw/
    8000.xz
    8000.hdf5
    README.txt             # Zenodo key list
    DupMols.dat            # duplicated equilibrium ids
  shards/8000/…            # only after convert_shard / convert_all_shards
```

Then:

```bash
bash etc/qm7x_zenodo/train/train.sh --data-dir ~/data/qm7x
bash etc/qm7x_zenodo/inspect/run.sh list_hdf5 ~/data/qm7x/raw/8000.hdf5
```

Pass extra flags through to the Python converter after `--`, e.g.
`-- --seed 1 --hdf5 /already/decompressed.hdf5`.
