# Checkpoint-parity: loading a real lorem-jax checkpoint into metatrain

Validates
[`metatrain.experimental.lorem.modules.jax_parity`](https://github.com/EricBoittier/metatrain/blob/experimental/lorem/src/metatrain/experimental/lorem/modules/jax_parity.py)
(`JaxParityBackbone` / `JaxParityLongRange`) against a real, shipped
[`lorem-tmlr-archive`](https://github.com/sirmarcel/lorem-tmlr-archive)
checkpoint, rather than only the in-repo synthetic-checkpoint unit test
(`test_jax_parity_checkpoint_loader_round_trips`). Two steps, two
environments — same split as the rest of `etc/lorem-parity/`: JAX for
anything that touches `lorem-jax`/`marathon`/`flax`, the shared metatrain
venv for everything torch.

## Setup

```bash
git clone https://github.com/sirmarcel/lorem-tmlr-archive
```

You need a `lorem-jax` venv (see the top-level `README.md`'s "Reference
implementations" section, or `lorem-jax/README.md` — `pip install .` in a
Python >= 3.11 venv) and the shared metatrain venv from
`setup-metawork.sh`, both already set up in this workspace.

## 1. Dump a checkpoint + JAX reference (JAX venv)

```bash
.venv-lorem-jax/bin/python etc/lorem-parity/jax_checkpoint_parity/dump_reference.py \
    lorem-tmlr-archive/evals/AuMgO/lorem/run/checkpoints/R2_E+F \
    lorem-tmlr-archive/datasets/AuMgO_valid.xyz \
    --n-frames 20 --out aumgo_reference.npz
```

Writes `aumgo_reference.npz`: the checkpoint's full flax parameter tree,
flattened to plain numpy arrays (no JAX needed to read it back), plus JAX
-computed energies/forces on the first 20 frames of `AuMgO_valid.xyz` using
the shipped checkpoint via `lorem.calculator.Calculator.from_checkpoint`
(the same call the paper's own `evals/*/metrics_*.py` scripts use).

## 2. Load into metatrain and compare (metatrain venv)

```bash
.venv/bin/python etc/lorem-parity/jax_checkpoint_parity/compare.py \
    lorem-tmlr-archive/evals/AuMgO/lorem/run/checkpoints/R2_E+F/model/model.yaml \
    lorem-tmlr-archive/datasets/AuMgO_valid.xyz \
    aumgo_reference.npz
```

Builds `JaxParityBackbone`/`JaxParityLongRange` with hypers read straight
from the checkpoint's own `model.yaml` (falling back to `lorem.Lorem`'s
field defaults for anything `model.yaml` doesn't override), loads the
dumped checkpoint via `jax_parity_checkpoint.load_checkpoint`, and reports
energy/force RMSE against the JAX reference from step 1. On AuMgO this
should print something close to:

```
energy: RMSE=0.0002 MAE=0.0001 meV/atom (torch port vs. JAX reference)
forces: RMSE=0.03 MAE=... meV/Angstrom
```

— i.e. exact agreement (float32 noise floor), not an approximation. Same on
bio_dimers (energy RMSE 0.243 vs. the paper's own 0.222 meV/atom, forces
1.77 vs. 1.65 meV/Å — within noise of the paper's *own* reported numbers).

## Known caveats

- **Ewald `smearing`/`kspace_resolution` are derived from `cutoff`, not read
  from the checkpoint** — lorem-jax's own `marathon.prepare()` computes
  `smearing = cutoff / 4`, `lr_wavelength = cutoff / 8` at data-prep time,
  and that choice isn't recorded in `model.yaml`. `compare.py` reproduces
  this formula directly (see `jax_parity.py`'s module docstring) rather than
  using `torchpme.tuning.ewald.tune_ewald`'s own recommendation — the two
  give numerically indistinguishable *converged* Ewald sums (verified
  against an analytic Madelung-constant ground truth), but only marathon's
  specific choice matches what a given checkpoint was actually trained
  against, in case that ever matters for something smearing-sensitive.
- **Non-PBC checkpoints** (`NaCl`/`cumulene`/`sn2`, excluding the `-nolr-*`
  and `-sr-*` variants) need the archive's documented ×2 long-range-potential
  correction to match their *shipped* numbers exactly — see the top-level
  archive `README.md`'s "Non-PBC LR Coulomb convention" section. Not applied
  here; irrelevant if you're training fresh rather than reproducing the
  paper-era checkpoint bit-for-bit.

### The bug that was here, for the record

Earlier versions of this checkpoint-parity work had a real, large, and
persistent discrepancy on periodic checkpoints (~0.7 meV/atom energy on
AuMgO, much worse on bio_dimers) that looked exactly like a numerical
disagreement between `torch-pme` and `jax-pme`'s Ewald implementations.
It wasn't: `JaxParityLongRange` was constructing its `torchpme.CoulombPotential`
with `exclusion_radius=neighbor_list_options.cutoff`, copied from the
*production* `LoremLongRangeFeaturizer` (which deliberately restructures
that split — see that class's own docstring). lorem-jax's real `Ewald()`
factory (`jaxpme.batched_mixed.calculators`) always uses
`exclusion_radius=None` — plain, unmodified Ewald. Fixed by removing the
`exclusion_radius` argument entirely.

Finding it required ruling out the library-numerics explanation with actual
ground truth rather than just comparing the two packages to each other:
building a NaCl rock-salt lattice with the exact, independently-known
analytic Madelung energy and confirming both `torch-pme` and `jax-pme`
reproduce it exactly (cubic and slab-like elongated cells, neutral and
non-neutral charges, with and without an exclusion radius) — at which point
"the libraries disagree" was no longer tenable, and hooking the *actual*
lorem-jax forward pass (`jax.debug.callback` on `Ewald()`'s returned
`potentials` closure, not an external reconstruction) to capture its true
inputs/outputs pointed straight at the one differing constructor argument.

## Warm-started fine-tuning

`train_smoke.py` warm-starts a `JaxParityBackbone`/`JaxParityLongRange` pair
from a dumped checkpoint and runs a short Adam fine-tune on the archive's
own training data, to check how close a warm start + brief training gets to
the paper's reported accuracy (not just the raw checkpoint-loading parity
above). On `cumulene`, 200 epochs on the full 2000-frame training set
(non-PBC, ~2.2h on CPU) reached:

| | energy RMSE | forces RMSE |
| --- | --- | --- |
| paper (shipped checkpoint, full test set) | 3.31 meV/atom | 50.1 meV/Å |
| this run | 3.85 meV/atom | 59.6 meV/Å |

— within ~15-20% of paper accuracy from a fraction of the paper's own
training budget (they use 2000 epochs with a LAMB optimizer; this used plain
Adam). See `train_smoke.py` for the training loop, loss weighting, and LR
schedule.

That cumulene run predates the `exclusion_radius` fix above (cumulene is
non-PBC, so it exercises `direct_calculator`, not `EwaldCalculator`, but the
same wrong argument was set there too) — expect it to do at least as well,
plausibly noticeably better, on a rerun. Not rerun here; the periodic
checkpoints (AuMgO, bio_dimers) were the priority once the bug was found.
