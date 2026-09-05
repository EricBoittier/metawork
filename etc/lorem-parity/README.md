# LOREM train / eval (outside the metatrain PR)

Optional recipes to run **experimental.lorem** next to the official JAX
examples. This folder is not part of metatrain CI.

`experimental.lorem` is a metatrain-native port. Same equations and knobs
as the paper / lorem-jax; **no bit-exact energy match** (different SH,
radial basis, PME, RNG). Contract tests live in
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
  `pip install 'metatrain[lorem]'` (or the tox `lorem-tests` env).
- **JAX / lorem-jax** — a separate venv; follow
  [`lorem-jax/README.md`](../../lorem-jax/README.md).

iris PETLR is a different trunk (PET + scalar charges). It is documented
in the architecture README; there is no iris recipe here.

## experimental.lorem (`mtt`)

Two epochs on the carbon subset already used by the architecture tests.
Run from the **metawork root**:

```bash
mtt train etc/lorem-parity/options.yaml
```

Data: [`metatrain/tests/resources/carbon_reduced_100.xyz`](../../metatrain/tests/resources/carbon_reduced_100.xyz).
Hypers are paper-shaped but small (`max_degree: 1`, 16 features) so the
run is short.

## lorem-jax (pointers only)

No vendored JAX. Use the upstream examples in their own venv:

- energy: [`lorem-jax/examples/train-mlp`](../../lorem-jax/examples/train-mlp)
- BEC / APT: [`lorem-jax/examples/train-bec`](../../lorem-jax/examples/train-bec)

If you want numbers, run both trainers and look. There is no energy-diff
script.
