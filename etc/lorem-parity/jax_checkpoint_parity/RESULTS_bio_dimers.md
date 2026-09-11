# bio_dimers checkpoint-parity results

Checkpoint: [`evals/bio_dimers/lorem/run/checkpoints/R2_E+F`](https://github.com/sirmarcel/lorem-tmlr-archive/tree/main/evals/bio_dimers/lorem/run/checkpoints/R2_E%2BF)
(periodic, 30×30×30 Å box, Ewald). 200 frames from `bio_dimers_test.xyz`
(13,743 total; first 200 used here), 3,891 atoms total.

Two different things are being measured, and it matters not to conflate
them:

1. **Checkpoint-loading fidelity** — does the torch port (`JaxParityBackbone`
   / `JaxParityLongRange`, loaded via `jax_parity_checkpoint.load_checkpoint`)
   reproduce the *same checkpoint's* predictions as running it in JAX? This
   is a weight-transfer/numerics check, nothing to do with the model's
   actual quality.
2. **Model accuracy vs. DFT** — how good is that checkpoint's energy/force
   prediction against the real DFT labels in the dataset? This is the
   number the paper itself reports (0.222 meV/atom energy RMSE, 1.646 meV/Å
   forces RMSE, full test set) — a property of the trained model, not of
   this port.

## Results

| | energy RMSE | energy MAE | forces RMSE | forces MAE |
| --- | --- | --- | --- | --- |
| **torch port vs. JAX reference** (checkpoint fidelity) | 0.000108 meV/atom | 0.000069 | 0.00271 meV/Å | 0.00059 |
| **torch port vs. DFT** (model accuracy) | 0.2415 meV/atom | 0.1751 | 1.7325 meV/Å | 1.0840 |
| paper (shipped checkpoint, full 13,743-frame test set) | 0.222 meV/atom | 0.155 | 1.646 meV/Å | 1.023 |

Checkpoint fidelity is at the float32 noise floor — this is what "the
port is correct" looks like, not an approximation. Model accuracy on this
200-frame slice is within ~9% (energy) / ~5% (forces) of the paper's own
number on the *full* 13,743-frame test set; the difference is consistent
with sampling only ~1.5% of the test set, not a systematic gap.

See `bio_dimers_parity_report.pdf` (regenerate with the commands below) for
per-frame residual plots and force-error histograms for both comparisons.

## How to reproduce this exact result

From the `metawork` root, with `lorem-tmlr-archive` cloned alongside:

```bash
# 1. JAX venv: dump the checkpoint + JAX reference energies/forces
.venv-lorem-jax/bin/python etc/lorem-parity/jax_checkpoint_parity/dump_reference.py \
    lorem-tmlr-archive/evals/bio_dimers/lorem/run/checkpoints/R2_E+F \
    lorem-tmlr-archive/datasets/bio_dimers_test.xyz \
    --n-frames 200 --out bio_dimers_reference_200.npz

# 2. metatrain venv: load into the torch port and compare
.venv/bin/python etc/lorem-parity/jax_checkpoint_parity/compare.py \
    lorem-tmlr-archive/evals/bio_dimers/lorem/run/checkpoints/R2_E+F/model/model.yaml \
    lorem-tmlr-archive/datasets/bio_dimers_test.xyz \
    bio_dimers_reference_200.npz
```

`compare.py`'s printed numbers are the "vs. JAX reference" row above. For
both rows plus the plots in `bio_dimers_parity_report.pdf`, use
`plot_parity.py` instead (same first two arguments, plus the dumped
`.npz` and an `--out` path):

```bash
.venv/bin/python etc/lorem-parity/jax_checkpoint_parity/plot_parity.py \
    lorem-tmlr-archive/evals/bio_dimers/lorem/run/checkpoints/R2_E+F/model/model.yaml \
    lorem-tmlr-archive/datasets/bio_dimers_test.xyz \
    bio_dimers_reference_200.npz \
    --out bio_dimers_parity_report.pdf
```

## What this validates (and a bug this exercise caught)

This result depends on the `exclusion_radius=None` fix to
`JaxParityLongRange`'s Ewald calculator (see `jax_parity.py`'s module
docstring and the "bug that was here" note in this directory's main
`README.md`). Before that fix, this exact command gave energy RMSE
**26.9 meV/atom** vs. JAX and forces RMSE **225 meV/Å** — both fixed by
one constructor argument, not a training or weight-loading problem. Also
depends on using marathon's actual smearing formula
(`smearing = cutoff/4 = 1.25`, `lr_wavelength = cutoff/8 = 0.625` for this
checkpoint's `cutoff=5.0`) rather than an independently-tuned guess —
`compare.py` computes this from `model.yaml` directly, so nothing to set
by hand.
