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
(dipole head) rather than a stale tox install.

```bash
bash etc/sn2_zenodo/convert.sh          # 200 random structures → ~/data/sn2
bash etc/sn2_zenodo/train.sh            # 5-epoch energy + forces + dipole
```

| file | role |
| --- | --- |
| [`convert.py`](convert.py) | download `sn2_reactions.npz`, write `sn2.xyz` |
| [`convert.sh`](convert.sh) | wrapper (`--n-samples all` for the full set) |
| [`options/energy-forces-dipole-lorem.yaml`](options/energy-forces-dipole-lorem.yaml) | `mtt train` |
| [`eval.yaml`](eval.yaml) | `mtt eval` on the same XYZ |
| [`train.sh`](train.sh) | train then eval from `~/data/sn2` |

The XYZ key for the dipole is ``dipole_moment`` so ASE does not park it
on the calculator (reserved name ``dipole``). Units: energy eV
(atomization), forces eV/Å, dipole e·Å wrt the origin. Total charge is
in ``info["charge"]`` but is not a training target.

``convert.py`` keeps structures with at least 5 atoms by default (CH₃X
and the six-atom SN2 complexes). Pass ``--min-atoms 0`` to include
HX / XY / CHX fragments.

```bash
python etc/sn2_zenodo/convert.py --n-samples 1000 --data-dir ~/data/sn2
DATA_DIR=~/data/sn2 bash etc/sn2_zenodo/train.sh
# mapping smoke test (no download):
pytest etc/sn2_zenodo/test_convert.py
```

Cite the PhysNet paper and the Zenodo DOI when you use the numbers.
