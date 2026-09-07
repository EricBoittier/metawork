# SN2 reactions (Zenodo) → experimental.lorem

[PhysNet SN2 dataset](https://zenodo.org/records/2605341)
(Unke & Meuwly, [arXiv:1902.08408](https://arxiv.org/abs/1902.08408);
DOI [10.5281/zenodo.2605341](https://doi.org/10.5281/zenodo.2605341)):
X⁻ + CH₃Y → CH₃X + Y⁻ (X,Y = F, Cl, Br, I) plus fragments. About 452k
gas-phase structures with **energy, forces, and dipole** at
DSD-BLYP-D3(BJ)/def2-TZVP.

This folder is **not** metatrain CI. The in-repo one-step tests use
synthetic dipoles; this recipe downloads the real set. ``train.sh``
puts ``metatrain/src`` on ``PYTHONPATH`` so ``mtt`` uses the checkout
(dipole head + isolated-atom force-grad sanitize) rather than a stale
tox install. After a ``max_degree > 2`` train it also passes ``-e extensions``
to ``mtt eval`` (sphericart TorchScript op).

```bash
bash etc/sn2_zenodo/convert.sh          # 10000 random structures → ~/data/sn2
# From scratch (required after architecture changes; see below):
mv ~/data/sn2/model.ckpt ~/data/sn2/model-degree2.ckpt   # if an old ckpt exists
RESTART=0 bash etc/sn2_zenodo/train.sh
# Compatible restart only (same max_degree / trunk / widths as the ckpt):
bash etc/sn2_zenodo/train.sh
# PET short-range trunk (optional pet.pretrained → PET-MAD):
YAML=etc/sn2_zenodo/options/energy-forces-dipole-lorem-pet.yaml \
  RESTART=0 bash etc/sn2_zenodo/train.sh
```

| file | role |
| --- | --- |
| [`convert.py`](convert.py) | download `sn2_reactions.npz`, write `sn2.xyz` |
| [`convert.sh`](convert.sh) | wrapper (`--n-samples all` for the full set) |
| [`options/energy-forces-dipole-lorem.yaml`](options/energy-forces-dipole-lorem.yaml) | `mtt train` (spherical short-range) |
| [`options/energy-forces-dipole-lorem-pet.yaml`](options/energy-forces-dipole-lorem-pet.yaml) | `mtt train` (PET short-range; optional PET-MAD init) |
| [`eval.yaml`](eval.yaml) | `mtt eval` on the same XYZ |
| [`train.sh`](train.sh) | train then eval from `~/data/sn2` |
| [`plot_reaction_coordinate.py`](plot_reaction_coordinate.py) | ASE POV-Ray snapshots on a shared-ξ matplotlib figure |

The XYZ key for the dipole is ``dipole_moment`` so ASE does not park it
on the calculator (reserved name ``dipole``). Units: energy eV
(atomization), forces eV/Å, dipole e·Å wrt the origin. Total charge is
in ``info["charge"]`` but is not a training target.

``convert.py`` keeps structures with at least 5 atoms by default (CH₃X
and the six-atom SN2 complexes). Pass ``--min-atoms 0`` to include
HX / XY / CHX fragments.

```bash
python etc/sn2_zenodo/convert.py --n-samples 10000 --data-dir ~/data/sn2
DATA_DIR=~/data/sn2 bash etc/sn2_zenodo/train.sh
# mapping smoke test (no download):
pytest etc/sn2_zenodo/test_convert.py
# Cl–Br reaction coordinate with POV-Ray annotations (needs povray + ASE):
metatrain/.tox/lorem-tests/bin/python etc/sn2_zenodo/plot_reaction_coordinate.py
```

A first 50-epoch CPU run on 1000 structures (800/100/100, 93k parameters,
``max_degree: 2``) does learn. The recipe default is now a **10000**-structure
subset (8000/1000/1000) and a larger spherical model (``max_degree: 6``,
``num_spherical_features: 16``, ``num_message_passing: 2``, ~132k parameters).

``train.sh`` restarts from ``~/data/sn2/model.ckpt`` only when that
checkpoint's model hypers match the yaml. Otherwise it **refuses**: this
checkout's ``mtt train --restart`` keeps the checkpoint architecture and
ignores yaml (upstream metatrain [#1232](https://github.com/metatensor/metatrain/pull/1232)
is not in ``experimental/lorem`` yet). Train the current yaml from scratch:

```bash
mv ~/data/sn2/model.ckpt ~/data/sn2/model-degree2.ckpt
cd ~/data/sn2
PYTHONPATH=/path/to/metawork/metatrain/src \
  /path/to/metawork/metatrain/.tox/lorem-tests/bin/mtt train \
  /path/to/metawork/etc/sn2_zenodo/options/energy-forces-dipole-lorem.yaml
# or:
RESTART=0 bash etc/sn2_zenodo/train.sh
```

``num_epochs: 50`` is a full run, not "50 more on top of the old 92k ckpt".
``learning_rate: 0.0001`` and ``scheduler_factor: 0.8``.

| | energy (meV/atom) | forces (meV/Å) | dipole (e·Å/atom) |
| --- | ---: | ---: | ---: |
| val, epoch 0 | 128 | 1158 | 0.162 |
| test, best | 33 | 268 | 0.042 |

Cite the PhysNet paper and the Zenodo DOI when you use the numbers.
