"""Instruments `LoremLongRangeFeaturizer.map_charges` (production LOREM long-range
module, `metatrain.experimental.lorem.modules.long_range`) to capture the per-atom
"charges" it feeds to its Ewald/P3M/direct Coulomb calculator.

Built to answer one question raised by notebook 17/20: is the huge ASE (non-periodic)
vs. i-PI/LAMMPS (periodic) energy disagreement for `sn2-matched-lorem` actually caused
by the *physical* net charge of the test structure (as notebook 17 assumed), or by
something else? The answer, checked here: `map_charges`'s scalar-charge channel is a
plain MLP over local invariant features with **no charge-neutrality or total-charge
constraint anywhere** -- its per-system sum is neither ~0 for a neutral molecule nor
close to the true net charge for a charged one (see `test_lorem_ewald_parity.py`).
That unconstrained, essentially-always-nonzero monopole is what a periodic Ewald
evaluation sees interacting with its own images across the box; a non-periodic
evaluation never sees it. Making the *test structure* neutral (notebook 20) does not
fix this, because the model's own internal charge prediction was never neutral to
begin with, for either structure.
"""

from typing import Dict, List

import torch


def load_lorem_eager(checkpoint_path: str):
    """Load a `sn2-matched-lorem`-style checkpoint (`model.ckpt`, not the exported
    `model.pt`) as a plain eager `torch.nn.Module`, so its submodules can still be
    monkeypatched/hooked -- the exported/scripted `model.pt` used by
    `hourglass_engines.py` cannot be, once traced/scripted.
    """
    import sphericart.torch  # noqa: F401  (registers torch extension ops)
    import metatomic.torch  # noqa: F401  (registers ModelMetadata etc. for torch.load)
    from metatrain.experimental.lorem.model import LOREM

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LOREM.load_checkpoint(ckpt, context="export")
    model.eval()
    return model


def capture_charges(model, atoms_list) -> List[torch.Tensor]:
    """Run `model` (as exported to an in-memory, non-scripted `AtomisticModel` via
    `MetatomicCalculator`) on each of `atoms_list`, returning the `map_charges`
    output captured for each call, in order -- `charges[:, 0]` is the scalar charge
    channel; `charges[:, 1:]` are the spherical (dipole/quadrupole/...) channels.
    """
    from metatomic_ase import MetatomicCalculator
    from metatrain.experimental.lorem.modules.long_range import LoremLongRangeFeaturizer

    captured: List[torch.Tensor] = []
    original = LoremLongRangeFeaturizer.map_charges

    def patched(self, features, spherical_features):
        charges = original(self, features, spherical_features)
        captured.append(charges.detach().clone())
        return charges

    LoremLongRangeFeaturizer.map_charges = patched
    try:
        exported = model.export()
        calc = MetatomicCalculator(exported)
        for atoms in atoms_list:
            a = atoms.copy()
            a.calc = calc
            a.get_potential_energy()
    finally:
        LoremLongRangeFeaturizer.map_charges = original

    return captured


def scalar_charge_sums(model, atoms_list) -> List[float]:
    """Convenience wrapper: total (summed over atoms) predicted scalar-charge
    channel for each structure in `atoms_list`."""
    charges = capture_charges(model, atoms_list)
    return [float(c[:, 0].sum()) for c in charges]
