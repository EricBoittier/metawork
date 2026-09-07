# Matched-budget SN2 comparison

Six recipes (LOREM / PET / SOAP-BPNN, each with and without the dipole
target) built to isolate one variable: **long-range electrostatics**.
Everything else that's easy to control is held equal across all six runs.

## What's controlled

| | LOREM | PET | SOAP-BPNN |
|---|---|---|---|
| Parameters (energy+forces) | 92.8K | 96.4K | 95.5K |
| Parameters (+dipole) | 93.0K | 103K | 96.1K |
| Cutoff radius | 5.0 A | 5.0 A | 5.0 A |
| Learning rate | 1e-4 | 1e-4 | 1e-4 |
| LR schedule | cosine, 1% warmup | cosine, 1% warmup | cosine, 1% warmup |
| `best_model_metric` | `rmse_prod` | `rmse_prod` | `rmse_prod` |
| `num_epochs` | 50 | 50 | 50 |
| `batch_size` | 16 | 16 | 16 |
| `scale_targets` | true | true | true |
| Train/val/test split | same `seed: 42`, same fractions -> identical split across all six |

Parameter counts aren't exactly equal -- each architecture's hyperparameters
are discrete (layer counts, `max_angular`/`max_radial`, embedding dims), so
"~93-103K" is as tight as a grid search by hand gets in reasonable time. The
dipole variants are each a bit larger than their energy+forces sibling
because the extra target needs its own head; PET's head is relatively the
most expensive of the three (+11K vs. LOREM's +0.2K and SOAP-BPNN's +0.5K),
which is itself a real (if small) architectural difference, not a mistake.

**LOREM's cosine schedule required a code change**: prior to this recipe
set, `experimental.lorem`'s trainer only supported `ReduceLROnPlateau`
(config: `scheduler_factor`/`scheduler_patience`), which has no equivalent
in PET/SOAP-BPNN. Added a `scheduler: "plateau" | "cosine"` hyperparameter
to `experimental.lorem` (default remains `"plateau"`, so existing configs
are unaffected) that reuses PET/SOAP-BPNN's exact warmup+cosine-decay
`LambdaLR` schedule when set to `"cosine"`, as used here.

## What's deliberately NOT controlled

- **Long-range electrostatics**: LOREM runs with Ewald long-range enabled
  (`long_range.enable: true`); PET and SOAP-BPNN are both short-range only
  (`long_range.enable: false`, PET explicitly, SOAP-BPNN by default). This
  is the comparison these recipes exist to make -- does long-range help,
  architecture capacity and training budget held equal -- not something to
  equalize away.
- Everything about *how* each architecture turns a cutoff-radius
  neighborhood into features (SOAP power spectrum vs. PET's graph
  transformer vs. LOREM's spherical-harmonic backbone) is, of course, still
  different -- that's what "architecture" means here.

## Running

```bash
bash etc/sn2_zenodo/convert.sh  # once, if not already done
YAML=etc/sn2_zenodo/options/matched-budget/energy-forces-lorem.yaml \
  DATA_DIR=~/data/sn2-matched-lorem bash etc/sn2_zenodo/train.sh
# ...same pattern for the other five, with distinct DATA_DIR values.
```
