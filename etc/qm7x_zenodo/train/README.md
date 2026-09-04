# Train on converted QM7-X

[`../options/`](../options/) holds one YAML per endpoint × architecture.
These scripts run `mtt train` in a per-job folder so checkpoints do not
overwrite each other.

Convert first (writes `qm7x.xyz` + `polarizability_spherical.mts`):

```bash
bash etc/qm7x_zenodo/convert/convert.sh
```


## Scripts

| script | what it does |
| --- | --- |
| [`train.sh`](train.sh) | one YAML (default `multi-pet`) |
| [`train_all.sh`](train_all.sh) | every matching YAML; keeps going on failure |

```bash
# six endpoints at once, PET (the default)
bash etc/qm7x_zenodo/train/train.sh

# one endpoint
bash etc/qm7x_zenodo/train/train.sh energy-pet
bash etc/qm7x_zenodo/train/train.sh dipole-soap_bpnn
bash etc/qm7x_zenodo/train/train.sh polarizability-spherical-pet

# a shard you converted separately
bash etc/qm7x_zenodo/train/train.sh --data-dir ~/data/qm7x/shards/8000 energy-pet

# every PET YAML
bash etc/qm7x_zenodo/train/train_all.sh --model pet

# every energy YAML (pet, soap_bpnn, gap, mace, space, dpa3)
bash etc/qm7x_zenodo/train/train_all.sh --endpoint energy
```

`multi-zip-pet` is skipped unless `qm7x.zip` is in the data directory
(`convert.sh --format zip`).


## Where files land

```
~/data/qm7x/
  qm7x.xyz
  polarizability_spherical.mts
  runs/multi-pet/model.pt
  runs/multi-pet/outputs/          # checkpoints, logs
  runs/energy-pet/…
```

Each run directory gets symlinks to the XYZ / `.mts` / zip so the YAML
paths (`read_from: qm7x.xyz`) resolve without copying the data.


## What the default venv can actually train

`setup-metawork.sh` installs metatrain with the `soap-bpnn,pet` extras.
Those two architectures will run as-is. The others need extras that are
**not** in the shared venv by default (they pull heavier or pinned
dependencies):

| filename suffix | `architecture.name` | default venv |
| --- | --- | --- |
| `-pet` | `pet` | yes |
| `-soap_bpnn` | `soap_bpnn` | yes |
| `-gap` | `gap` | no (`gap` extra) |
| `-mace` | `experimental.mace` | no (`mace` extra) |
| `-space` | `experimental.space` | no |
| `-dpa3` | `experimental.dpa3` | no (`dpa3` extra; also pins torch) |

`train_all.sh` records failures and continues, so a missing extra does
not abort the rest. Use `--model pet` or `--model soap_bpnn` to stay
inside what is installed.


## Hypers are demos

Every YAML has `num_epochs: 5` and a small `batch_size`. That is enough
to check that the `TensorMap` layout trains; it is not a production fit.
Raise epochs (and the architecture `model:` block) in a copy of the YAML,
or override on the CLI:

```bash
cd ~/data/qm7x/runs/energy-pet
mtt train /path/to/metawork/etc/qm7x_zenodo/options/energy-pet.yaml \
  -o model.pt \
  -r architecture.training.num_epochs=50
```

See [`../options/README.md`](../options/README.md) for which models
support which tensor types, and the
[generic-targets tutorial](https://docs.metatensor.org/metatrain/latest/generated_examples/1-advanced/03-fitting-generic-targets.html).
