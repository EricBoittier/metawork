# QM7-X from Zenodo → metatensor → multi-endpoint metatrain

This folder reads the **original** QM7-X HDF5 dump from
[Zenodo record 4288677](https://zenodo.org/records/4288677)
([Hoja *et al.*, *Sci. Data* **8**, 43 (2021)](https://www.nature.com/articles/s41597-021-00812-2)),
not the OpenQDC energy/force wrapper. The HDF5 files store ~40
physicochemical properties per structure; the point of this example is to
turn several of those, with **different sample kinds and tensor characters**,
into `TensorMap`s and train them jointly with metatrain.

For the OpenQDC (energy + forces only) path, see
[`../openqdc_metatomic/`](../openqdc_metatomic/).

```
Zenodo 8000.xz  →  8000.hdf5  →  TensorMap(s) + System(s)
                                      │
                         xyz + polarizability_spherical.mts
                         or  qm7x.zip (DiskDataset)
                                      │
                    mtt train etc/qm7x_zenodo/options/<endpoint>-<model>.yaml
```

Runnable wrappers (data dir defaults to `~/data/qm7x`):

| folder | purpose |
| --- | --- |
| [`convert/`](convert/) | download a shard and write `qm7x.xyz` |
| [`train/`](train/) | `mtt train` one YAML, or every matching YAML |
| [`inspect/`](inspect/) | HDF5 key catalog, dump one conformation, summarize XYZ |
| [`options/`](options/) | one YAML per endpoint × architecture |

```bash
bash etc/qm7x_zenodo/convert/convert.sh          # 200 structures from 8000.xz
bash etc/qm7x_zenodo/train/train.sh              # multi-pet, 5 epochs
bash etc/qm7x_zenodo/train/train.sh energy-pet
bash etc/qm7x_zenodo/convert/convert_all_shards.sh   # all 8 shards, 200 each
```


## What lives in the HDF5

Each shard (`1000.xz` … `8000.xz`) decompresses to an HDF5 file whose groups
are molecule IDs, then conformation IDs (`Geom-mr-is-ct-u`). Properties are
datasets on the conformation group — see the
[Zenodo README](https://zenodo.org/records/4288677/files/README.txt?download=1).
The shard this example downloads by default is `8000.xz` (~89 MB), the file
that README recommends for a first pass.

| HDF5 key | unit | layout | this example |
| --- | --- | --- | --- |
| `atNUM` / `atXYZ` | — / Å | `(N,)` / `(N, 3)` | geometry |
| `eAT` | eV | scalar | `energy` (atomization; forces identical to total energy) |
| `totFOR` | eV/Å | `(N, 3)` | `energy` positions gradient |
| `HLgap` | eV | scalar | `mtt::hlgap` |
| `vDIP` | e·Å | `(3,)` | `mtt::dipole` |
| `hCHG` | e | `(N,)` | `mtt::hirshfeld_charge` |
| `hVDIP` | e·bohr | `(N, 3)` | `mtt::hirshfeld_dipole` |
| `mTPOL` | bohr³ | `(3, 3)` | `mtt::polarizability` (spherical λ=0,2) |
| `mC6` / `atC6` | Eh·a₀⁶ | scalar / `(N,)` | written, not in the default YAML |

The other ~30 keys (HOMO/LUMO, quadrupole, Hirshfeld volumes, KS eigenvalues, …)
follow the same pattern: pick a sample kind and a tensor type, wrap a
`TensorBlock`, save `.mts`. Full catalog:
[`inspect/hdf5-properties.md`](inspect/hdf5-properties.md).


## Endpoint shapes

These six targets are the interesting spread — they are **not** the same
`TensorMap` layout:

| target | sample kind | character | `TensorMap` layout |
| --- | --- | --- | --- |
| `energy` | system | scalar + `positions` gradient | 1 block, no components; gradient has `xyz` |
| `mtt::hlgap` | system | scalar | 1 block, shape `(n_sys, 1)` |
| `mtt::dipole` | system | Cartesian rank 1 | 1 block, component `xyz`, shape `(n_sys, 3, 1)` |
| `mtt::hirshfeld_charge` | atom | scalar | 1 block, samples `[system, atom]`, shape `(n_atoms, 1)` |
| `mtt::hirshfeld_dipole` | atom | Cartesian rank 1 | samples `[system, atom]`, component `xyz` |
| `mtt::polarizability` | system | spherical λ=0 and λ=2 | **two** blocks, keys `o3_lambda`/`o3_sigma`, component `o3_mu` |

ASE can store the first five (scalars in `atoms.info` / `atoms.arrays`,
vectors as length-3 arrays). It **cannot** store a spherical tensor with more
than one irrep — that is why polarizability is a sidecar `.mts` file (or a
member of the zip `DiskDataset`). A Cartesian rank-2 alternative
(`type.cartesian.rank: 2`, key `polarizability` in the XYZ) works with PET
but not with SOAP-BPNN; the spherical form is the one that matches how
metatrain's own QM7-X tests are written.

`dipole` is an ASE reserved name (like `energy` / `forces`): extxyz writes
`dipole="x y z"` but `ase.io.read` puts it on the calculator, not
`atoms.info`. Metatrain's ASE reader copies it back; vanilla `ase.io.read`
will not see `atoms.info["dipole"]`.

PET can train on all six. SOAP-BPNN can do energy, scalars, spherical tensors,
and Cartesian rank 1, but not rank 2.


## Convert

Needs `h5py`, which `etc/setup-metawork.sh` installs into the venv. The
wrappers in [`convert/`](convert/) write to `~/data/qm7x` by default:

```bash
bash etc/qm7x_zenodo/convert/convert.sh                    # 200 from 8000.xz
bash etc/qm7x_zenodo/convert/convert.sh --n-samples all    # whole shard
bash etc/qm7x_zenodo/convert/convert.sh --format zip
bash etc/qm7x_zenodo/convert/convert_shard.sh 1000
bash etc/qm7x_zenodo/convert/convert_all_shards.sh         # all shards, 200 each
```

Same thing by calling the Python converter yourself (writes into the
current directory):

```bash
# 200 random structures from the small 8000.xz shard (downloaded on first run)
python etc/qm7x_zenodo/zenodo_to_metatensor.py --n-samples 200

# already-decompressed HDF5
python etc/qm7x_zenodo/zenodo_to_metatensor.py --hdf5 /path/to/8000.hdf5

# a different shard, full DiskDataset zip
python etc/qm7x_zenodo/zenodo_to_metatensor.py \
    --file 1000.xz --n-samples 500 --format zip
```

`--format xyz` writes `qm7x.xyz` plus `polarizability_spherical.mts`.
`--format zip` writes `qm7x.zip` with

```
0/system.mta
0/energy.mts              # scalar + positions gradient
0/hlgap.mts
0/dipole.mts
0/hirshfeld_charge.mts
0/hirshfeld_dipole.mts
0/polarizability.mts      # spherical TensorMap
…
```

The spherical conversion of `mTPOL` uses the same λ=0 (trace) / λ=2
(symmetrized Cartesian) convention as
`metatrain/tests/cli/dump_spherical_targets.py`.


## Train

Option files live in [`options/`](options/) — one YAML per endpoint and
architecture. [`train/`](train/) runs `mtt train` in a per-job folder
under `$DATA_DIR/runs/<stem>/`.

```bash
bash etc/qm7x_zenodo/convert/convert.sh
bash etc/qm7x_zenodo/train/train.sh                         # multi-pet
bash etc/qm7x_zenodo/train/train.sh energy-pet
bash etc/qm7x_zenodo/train/train.sh dipole-soap_bpnn
bash etc/qm7x_zenodo/train/train_all.sh --model pet
```

Or invoke metatrain directly from the directory that holds `qm7x.xyz`:

```bash
python etc/qm7x_zenodo/zenodo_to_metatensor.py --n-samples 200
mtt train etc/qm7x_zenodo/options/energy-pet.yaml
mtt train etc/qm7x_zenodo/options/dipole-soap_bpnn.yaml
mtt train etc/qm7x_zenodo/options/polarizability-spherical-mace.yaml
mtt train etc/qm7x_zenodo/options/multi-pet.yaml
```

`--format zip` plus `options/multi-zip-pet.yaml` reads a DiskDataset instead.

Each `mtt::…` block declares `sample_kind` (`system` vs `atom`) and `type`
(`scalar`, `cartesian.rank`, or `spherical.irreps`), as in the
[generic-targets tutorial](https://docs.metatensor.org/metatrain/latest/generated_examples/1-advanced/03-fitting-generic-targets.html).

To add another HDF5 property, see [`inspect/README.md`](inspect/README.md)
and [`inspect/hdf5-properties.md`](inspect/hdf5-properties.md). Extend
`load_structure()` / `to_targets()` and append a target section. Examples:

```yaml
mtt::c6:                    # molecular C6 — per-structure scalar
  key: c6
  sample_kind: system
  type: scalar

mtt::atomic_c6:             # atomic C6 — per-atom scalar
  key: atomic_c6
  sample_kind: atom
  type: scalar
```

Those two can be added the same way (`info["c6"]` / `arrays["atomic_c6"]` are
already written to the XYZ when the HDF5 keys exist). Leave them out of a zip
run unless every structure has them — DiskDataset zips must be homogeneous.


## Building the `TensorMap`s yourself

The conversion script is the reference. The layouts in short:

```python
# per-structure scalar (HLgap, eAT, mC6, …)
TensorBlock(values=[[e]], samples=Labels("system", [[i]]),
            components=[], properties=Labels.single())

# per-atom scalar (hCHG, atC6, …)
TensorBlock(values=charges.reshape(n, 1),
            samples=Labels(["system", "atom"], [[i, a] for a in range(n)]),
            components=[], properties=Labels.single())

# per-structure vector (vDIP)
TensorBlock(values=dipole.reshape(1, 3, 1),
            samples=Labels("system", [[i]]),
            components=[Labels("xyz", [[0], [1], [2]])],
            properties=Labels.single())

# spherical polarizability: two blocks, keys (o3_lambda, o3_sigma) = (0,1), (2,1)
#   λ=0 values shape (n_sys, 1, 1), component o3_mu = [0]
#   λ=2 values shape (n_sys, 5, 1), component o3_mu = [-2, -1, 0, 1, 2]
```

Save with `TensorMap.save("name.mts")` and load from C/C++/Rust exactly as in
[`../openqdc_metatomic/README.md`](../openqdc_metatomic/README.md).


## References

- Dataset: [Zenodo 4288677](https://zenodo.org/records/4288677) ·
  [paper](https://www.nature.com/articles/s41597-021-00812-2) ·
  [HDF5 README](https://zenodo.org/records/4288677/files/README.txt?download=1)
- metatrain: [dataset formats](https://docs.metatensor.org/metatrain/latest/concepts/dataset-formats.html) ·
  [generic targets](https://docs.metatensor.org/metatrain/latest/generated_examples/1-advanced/03-fitting-generic-targets.html)
