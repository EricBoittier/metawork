# Training options (endpoint × model)

Each file is a complete `mtt train` input. Naming is `{endpoint}-{model}.yaml`.
Only combinations the architecture can actually train are included.

Convert a subset first:

```bash
bash etc/qm7x_zenodo/convert/convert.sh
bash etc/qm7x_zenodo/train/train.sh energy-pet
```

Or from the directory that will hold `qm7x.xyz`:

```bash
python etc/qm7x_zenodo/zenodo_to_metatensor.py --n-samples 200
mtt train etc/qm7x_zenodo/options/energy-pet.yaml
```

[`../train/`](../train/) wraps `mtt train` (per-job `runs/<stem>/`,
`--model` / `--endpoint` filters). Spherical polarizability also needs
`polarizability_spherical.mts` next to the XYZ. The zip multi file needs
`--format zip`.

Architecture names in YAML:

| short name in filename | `architecture.name` |
| --- | --- |
| `pet` | `pet` |
| `soap_bpnn` | `soap_bpnn` |
| `gap` | `gap` |
| `mace` | `experimental.mace` |
| `space` | `experimental.space` |
| `dpa3` | `experimental.dpa3` |

What each model can train (from the
[generic-targets tutorial](https://docs.metatensor.org/metatrain/latest/generated_examples/1-advanced/03-fitting-generic-targets.html)):

| | energy+forces | scalars | spherical | Cartesian |
| --- | --- | --- | --- | --- |
| PET | yes | yes | yes | rank 1 and 2 |
| SOAP-BPNN | yes | yes | yes | rank 1 only |
| MACE | yes | yes | yes | rank 1 only |
| SPACE | yes | yes | yes | rank 1 only |
| DPA3 | yes | yes | no | no |
| GAP | yes | no | no | no |

## Files

### Single endpoint

| file | target | layout |
| --- | --- | --- |
| `energy-{pet,soap_bpnn,gap,mace,space,dpa3}.yaml` | `energy` + forces | structure scalar + `positions` gradient |
| `hlgap-{pet,soap_bpnn,mace,space,dpa3}.yaml` | `mtt::hlgap` | structure scalar |
| `dipole-{pet,soap_bpnn,mace,space}.yaml` | `mtt::dipole` | structure Cartesian rank 1 |
| `hirshfeld-charge-{pet,soap_bpnn,mace,space,dpa3}.yaml` | `mtt::hirshfeld_charge` | atom scalar |
| `hirshfeld-dipole-{pet,soap_bpnn,mace,space}.yaml` | `mtt::hirshfeld_dipole` | atom Cartesian rank 1 |
| `polarizability-spherical-{pet,soap_bpnn,mace,space}.yaml` | `mtt::polarizability` | structure spherical λ=0,2 (`.mts`) |
| `polarizability-cartesian-pet.yaml` | `mtt::polarizability` | structure Cartesian rank 2 (XYZ) |
| `c6-pet.yaml` | `mtt::c6` | structure scalar (optional HDF5 key) |
| `atomic-c6-pet.yaml` | `mtt::atomic_c6` | atom scalar (optional HDF5 key) |

### Several endpoints at once

| file | contents |
| --- | --- |
| `multi-pet.yaml` | energy, gap, dipole, Hirshfeld charge/dipole, spherical polarizability |
| `multi-soap_bpnn.yaml` | same (no rank-2 Cartesian) |
| `multi-mace.yaml` | same |
| `multi-space.yaml` | same |
| `multi-dpa3.yaml` | energy + gap + Hirshfeld charge only |
| `multi-zip-pet.yaml` | same six as `multi-pet`, reading `qm7x.zip` |

Training hypers are intentionally short (`num_epochs: 5`). Raise that, and
the architecture `model:` block, for a real fit.
