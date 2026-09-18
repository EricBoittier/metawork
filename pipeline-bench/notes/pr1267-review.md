# Review notes for metatensor/metatrain#1267

Paste-ready. Not posted. The PR is
https://github.com/metatensor/metatrain/pull/1267.

`Trainer.setup()` is the right seam. These four are the cleanup I'd want
before building preprocessing on top of it.

## 1. `self.atomic_types` in ZBL setup is an AttributeError

In `Trainer.setup()`, composition is wired correctly:

```python
composition_model = CompositionModel.from_valid_targets(
    dataset_info, dataset_info.atomic_types
)
```

Two lines later the ZBL branch does:

```python
if model_hypers["zbl"]:
    ...
    ZBL(
        {},
        dataset_info=DatasetInfo(
            length_unit=model_dataset_info.length_unit,
            atomic_types=self.atomic_types,  # Trainer never sets this
            targets=zbl_targets,
        ),
    )
```

`Trainer` has `self.dataset_info` after this method starts, and never
`self.atomic_types`. That attribute lived on `PET`, and this was moved
without rewriting the receiver. Any PET run with `zbl: true` raises
`AttributeError` at setup, before a step runs.

Fix: `atomic_types=dataset_info.atomic_types` (or
`model_dataset_info.atomic_types`; they are the same list). A one-line
ZBL training test would have caught it; that path is untested in this PR.

## 2. Device intersection is computed and discarded

`MetatrainWrapper.export()`:

```python
supported_devices = set(all_supported_devices[0])
for devices in all_supported_devices[1:]:
    supported_devices.intersection_update(devices)

capabilities = ModelCapabilities(
    ...
    supported_devices=self.model.__supported_devices__,
)
```

The intersection of the exported components is the number that belongs
on the wrapper. `self.model.__supported_devices__` is the inner
architecture's class attribute, so a CPU-only additive (or scaler) would
still advertise CUDA. Use `sorted(supported_devices)`.

## 3. Atomic-basis sparsification is commented out

The wrapper's eval-time `forward` has the sparsify block commented, plus
a second commented `metatensor.torch.add` with a hand-rolled sparse sum
under it. This wrapper is supposed to be the inference composition;
leaving the basis path as comments means PET atomic-basis eval is either
wrong or still inside the architecture, which is the thing this PR is
trying to stop.

## 4. `WrapperHypers` does not describe what `setup()` passes

```python
class WrapperHypers(TypedDict):
    model: NotRequired[dict]
    additive_models: NotRequired[list[dict]]
    scaler: NotRequired[dict]
```

`Trainer.setup()` (and `export()`) pass live modules:

```python
hypers=dict(
    model=model,
    additive_models=additive_models,
    scaler=scaler,
)
```

The TypedDict is the public construction contract. Either it should say
`ModelInterface` / `ModuleList` / `Scaler`, or setup should not pretend
these are hypers dictionaries. As written it is prototype-level.

---

`Trainer.setup()` should keep declaring the training representation, not
grow into the place that also runs neighbor lists, caching, workers, and
device movement. Those stay in the data utilities. The wrapper should
stay the inference composition, not pick up training-pipeline mechanics.
