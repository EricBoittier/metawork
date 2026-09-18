# #1267 — four setup/wrapper bugs, and where they live now

Pol’s PR: https://github.com/metatensor/metatrain/pull/1267
Fixes: [`EricBoittier/metatrain:fix/1267-wrapper-setup`](https://github.com/EricBoittier/metatrain/tree/fix/1267-wrapper-setup)
(`ba4cb63d`, one commit on top of `pfebrer:mtt_wrapper`). Local checkout:
`/tmp/metatrain-1267`. Pushed. Not a PR. Compare:
https://github.com/pfebrer/metatrain/compare/mtt_wrapper...EricBoittier:metatrain:fix/1267-wrapper-setup

`Trainer.setup()` is still the right seam. These four were the cleanup
before building preprocessing on it. All four are done on the fix
branch; `test_setup_with_zbl` passed; ruff is clean.

| # | bug on `mtt_wrapper` | fix |
| --- | --- | --- |
| 1 | ZBL reads `self.atomic_types`; Trainer never sets it | `dataset_info.atomic_types` |
| 2 | export computes a device intersection, then ignores it | capabilities use that intersection, cuda-then-cpu |
| 3 | eval-time atomic-basis sparsify is commented out | wrapper sparsifies from `self.dataset_info` |
| 4 | `WrapperHypers` claims dicts; `setup()` passes modules | typed as live `ModelInterface` / list / `Scaler` |

The manual sparse add in the wrapper is still hand-rolled:
`metatensor.torch.add` still cannot sum missing keys. Left as a TODO,
not one of the four.

`Trainer.setup()` should keep declaring the training representation, not
grow into neighbor lists, caching, workers, or device movement. Those
stay in the data utilities. The wrapper stays the inference composition.

## Original notes (what was wrong on the PR)

### 1. `self.atomic_types` in ZBL setup is an AttributeError

Composition is wired correctly:

```python
composition_model = CompositionModel.from_valid_targets(
    dataset_info, dataset_info.atomic_types
)
```

Two lines later the ZBL branch did `atomic_types=self.atomic_types`.
That attribute lived on `PET` and was copied with the block. Any PET
run with `zbl: true` raised at setup.

### 2. Device intersection computed and discarded

`export()` built `supported_devices` as the intersection of component
capabilities, then passed `self.model.__supported_devices__` into
`ModelCapabilities`.

### 3. Atomic-basis sparsification commented out

The eval-time `forward` had the sparsify loop commented, so either
atomic-basis eval was wrong or it was still inside the architecture.

### 4. `WrapperHypers` did not describe what `setup()` passes

The TypedDict said `dict` / `list[dict]`; `setup()` and `export()`
pass live modules.
