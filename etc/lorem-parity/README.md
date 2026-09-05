# LOREM train / eval (outside the metatrain PR)

Optional recipes to run **experimental.lorem** next to the official JAX
examples. This folder is not part of metatrain CI.

`experimental.lorem` is a metatrain-native port. Same equations and knobs
as the paper / lorem-jax; **no bit-exact energy match** (PME
implementation, RNG, e3x cartesian vs ``m = -ℓ … +ℓ``). Contract tests live in
[`metatrain/src/metatrain/experimental/lorem/tests/test_paper_contracts.py`](../../metatrain/src/metatrain/experimental/lorem/tests/test_paper_contracts.py).
The comparison table is
[`metatrain/src/metatrain/experimental/lorem/README.md`](../../metatrain/src/metatrain/experimental/lorem/README.md).

## Setup

From the metawork root:

```bash
git submodule update --init
```

Use **two environments**. Do not pip-install JAX packages into the
metatrain venv from `setup-metawork.sh`.

- **PyTorch / mtt** — the shared metawork venv, with
  `pip install 'metatrain[lorem]'`, or the tox env:
  `metatrain/.tox/lorem-tests/bin/mtt`.
- **JAX / lorem-jax** — a separate venv; follow
  [`lorem-jax/README.md`](../../lorem-jax/README.md).

iris PETLR is a different trunk (PET + scalar charges). It is documented
in the architecture README; there is no iris recipe here.

## experimental.lorem (`mtt`)

Run from the **metawork root**. Hypers are paper-shaped but small
(`max_degree: 1`, 16 features) so the run is short.

Shared toy XYZ with the lorem-jax MLP example (same structures, not a
JAX energy match):

```bash
mtt train etc/lorem-parity/options-jax-toy.yaml
mtt eval model.pt etc/lorem-parity/eval.yaml -o output.xyz
```

`mtt` writes `outputs/`, `model.pt`, and `output.xyz` in the current
directory. Those names are gitignored at the repo root.

Carbon subset already used by the architecture tests:

```bash
mtt train etc/lorem-parity/options.yaml
```

| file | data |
| --- | --- |
| [`options-jax-toy.yaml`](options-jax-toy.yaml) | [`lorem-jax/examples/train-mlp/data.xyz`](../../lorem-jax/examples/train-mlp/data.xyz) (`forces`) |
| [`eval.yaml`](eval.yaml) | same XYZ, after export |
| [`options.yaml`](options.yaml) | [`metatrain/tests/resources/carbon_reduced_100.xyz`](../../metatrain/tests/resources/carbon_reduced_100.xyz) (`force`) |

## lorem-jax (pointers only)

No vendored JAX. Use the upstream examples in their own venv:

- energy (same `data.xyz` as `options-jax-toy.yaml`):
  [`lorem-jax/examples/train-mlp`](../../lorem-jax/examples/train-mlp)
- BEC / APT: [`lorem-jax/examples/train-bec`](../../lorem-jax/examples/train-bec)

```bash
cd lorem-jax/examples/train-mlp
DATASETS=. python prepare.py
cd my_experiment
DATASETS=.. lorem-train
```

## Architecture keys and comparison numbers

From the metawork root, with the PyTorch / mtt env:

```bash
metatrain/.tox/lorem-tests/bin/python etc/lorem-parity/compare.py
# or:  .../compare.py path/to/model.pt
```

That script:

1. Parses ``lorem.Lorem`` field names from the submodule source (no JAX
   import) and prints them next to ``experimental.lorem`` hypers keys.
2. Prints the torch ``named_parameters`` tree (``sr`` / ``lr`` scopes).
3. Scores a trained ``model.pt`` against the DFT labels on the same
   21-frame toy XYZ (energy / force RMSE and a per-frame table).

Default model path is the jax-toy run directory
``outputs/2026-09-05/10-21-57/model.pt``. Re-train if that is gone.

This is **fit-to-labels**, not ``E_mtt − E_jax``. Different spherical
harmonics, radial-basis implementation, PME, and independent RNG. The
in-repo CI test ``test_hypers_keys_overlap_lorem_jax`` prints the same
key table without touching this folder.

If you want numbers, run both trainers and look. There is no energy-diff
script.

Energy, forces, and dipoles on the [PhysNet SN2 set](https://zenodo.org/records/2605341)
(Zenodo 2605341) live in [`../sn2_zenodo/`](../sn2_zenodo/).
