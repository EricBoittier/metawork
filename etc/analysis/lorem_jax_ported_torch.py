"""A from-scratch torch port of the real lorem-jax SN2 checkpoint
(`lorem-tmlr-archive/evals/sn2/lorem/run/checkpoints/R2_E+F`, the ``lr: true``
variant) into ``JaxParityBackbone`` + ``JaxParityLongRange``
(``metatrain.experimental.lorem.modules.jax_parity``), wrapped as an
ASE-compatible calculator.

Those two torch modules are a research prototype: a module-for-module,
leaf-name-for-leaf-name port of lorem-jax's flax ``Lorem`` model, written so a
shipped lorem-jax checkpoint has an exact 1:1 home for every learned weight.
No full-checkpoint loader existed for them before this file -- this is the
first one, built and validated here.

Validation (see ``/home/boittier/analysis/15_long_range_scan_lorem_jax_ported_torch.ipynb``
for the full, live-captured report): energies/forces from this torch port are
compared against the real ``lorem.calculator.Calculator`` (jax) on the same
checkpoint, for `sn2/sn2.xyz` frame 0 and 9 points along the C...I
reaction-coordinate scan (1.5 to 32 Angstrom). After fixing two bugs found
during that validation (both documented on ``FixedJaxParityLongRange`` below),
agreement is:

    max |dE| = 1.99e-05 eV
    max |dF| = 3.42e-05 eV/Angstrom

i.e. float32-noise-level agreement, far inside the 1e-2 eV tolerance judgment
call in the task this module was built for.

Two bugs were found and fixed (both in *this* file only -- ``jax_parity.py``
itself is not modified):

1. ``JaxParityLongRange.__init__`` builds its non-PBC ``direct_calculator``
   with ``exclusion_radius=neighbor_list_options.cutoff`` (copy-pasted from
   the periodic ``ewald_calculator`` a few lines above). lorem-jax's actual
   non-PBC path (``jaxpme.batched_mixed.calculators.Ewald``'s ``real_space,
   no-pbc`` branch) is a plain, unrestricted ``sum_j q_j / r_ij`` over *all*
   pairs -- it has no near-field exclusion at all. With the cutoff-sized
   exclusion left in, every pair inside the model's 5.0 Angstrom SR cutoff
   (i.e. most of a small molecule) has its potential wrongly suppressed by
   torch-pme's exclusion cutoff function -- a large, structure-dependent
   error (checked directly against a 2-atom toy system's captured
   ``jaxpme`` potentials: with the cutoff-sized exclusion in place, the
   scalar potential channel came out ~9x too small and non-uniformly across
   channels, not a simple constant factor).
2. Once (1) is fixed, the plain-sum potential still needs an extra factor of
   2: the shipped non-PBC checkpoint was trained against
   ``V_i = sum_j q_j / r_ij`` (no double-counting factor), but *both*
   today's jax-pme and torch-pme's own ``Calculator`` return
   ``V_i = (1/2) sum_j q_j v(r_ij)`` by convention (see torch-pme's
   ``Calculator.forward`` docstring, and ``lorem-tmlr-archive/README.md``'s
   "Non-PBC LR Coulomb convention" section) -- this is the *same*
   after-the-fact convention change the archive's own README documents and
   that notebook 13 already patches on the jax reference side (``x2`` on
   ``Ewald().potentials``). It affects this torch port's non-PBC branch too
   and is applied here for the same reason.

The elemental baseline offset from ``model/baseline.yaml`` (``add_offset``
behavior in ``lorem.calculator.Calculator``) is NOT part of
``JaxParityBackbone``/``JaxParityLongRange`` -- it is added explicitly in
``JaxParityASECalculator.calculate`` below, exactly as
``lorem.calculator.Calculator.calculate`` does.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from ase.calculators.calculator import Calculator as _ASECalculator
from ase.calculators.calculator import all_changes

METATRAIN_SRC = "/home/boittier/metawork/metatrain/src"
if METATRAIN_SRC not in sys.path:
    sys.path.insert(0, METATRAIN_SRC)

from metatomic.torch import NeighborListOptions, System  # noqa: E402
from metatrain.experimental.lorem.modules import e3x_compat  # noqa: E402
from metatrain.experimental.lorem.modules.jax_parity import (  # noqa: E402
    JaxParityBackbone,
    JaxParityLongRange,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists  # noqa: E402

# -- hyperparameters, from lorem-tmlr-archive/evals/sn2/lorem/run/checkpoints/R2_E+F/model/model.yaml --
CUTOFF = 5.0
MAX_DEGREE = 6
MAX_DEGREE_LR = 2
NUM_FEATURES = 128
NUM_RADIAL = 32
NUM_SPHERICAL_FEATURES = 8
NUM_SPECIES = 8

# -- per-species energy offsets, from .../model/baseline.yaml's `elemental` key --
BASELINE_ELEMENTAL_OFFSETS: Dict[int, float] = {
    1: -3.5921101,
    6: -1.71485694,
    9: -0.97762923,
    17: -0.23682422,
    35: 0.08449515,
    53: 0.17498543,
}

DEFAULT_CHECKPOINT = (
    "/home/boittier/metawork/lorem-tmlr-archive/evals/sn2/lorem/run/"
    "checkpoints/R2_E+F"
)


# --------------------------------------------------------------------------
# flax msgpack -> flat dict of numpy arrays
# --------------------------------------------------------------------------


def read_checkpoint_params(checkpoint_dir: str) -> Dict[str, np.ndarray]:
    """Read a lorem-jax checkpoint's ``model.msgpack`` into a flat dict of
    ``flax/path/without/leading/params`` -> numpy array.

    This function must run in a Python environment with ``lorem``/``jax``
    installed (e.g. ``/home/boittier/metawork/.venv-lorem-jax/bin/python``),
    NOT the torch env this module otherwise expects -- see
    :func:`dump_checkpoint_params_to_npz`, which does the jax-side half of
    this split and can be invoked via ``subprocess`` from a torch process.
    """
    from lorem.calculator import read_msgpack

    path = Path(checkpoint_dir) / "model" / "model.msgpack"
    tree = read_msgpack(str(path))

    flat: Dict[str, np.ndarray] = {}

    def _walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{prefix}/{k}" if prefix else str(k))
        else:
            flat[prefix] = np.asarray(node)

    _walk(tree)
    # strip the top-level "params" wrapper, if present
    return {
        (k[len("params/") :] if k.startswith("params/") else k): v
        for k, v in flat.items()
    }


def dump_checkpoint_params_to_npz(checkpoint_dir: str, out_path: str) -> None:
    """jax-side helper: dump ``read_checkpoint_params`` to an ``.npz`` file
    that :func:`load_params_flat` (torch side) can read, bridging the two
    incompatible venvs via a file instead of importing both jax and torch in
    one process. Run this with the lorem-jax venv's python."""
    flat = read_checkpoint_params(checkpoint_dir)
    np.savez(out_path, **{k.replace("/", "__"): v for k, v in flat.items()})


def load_params_flat(npz_path: str) -> Dict[str, np.ndarray]:
    """Torch-side counterpart of :func:`dump_checkpoint_params_to_npz`."""
    data = np.load(npz_path)
    flat = {}
    for k in data.files:
        name = k.replace("__", "/")
        if name.startswith("params/"):
            name = name[len("params/") :]
        flat[name] = data[k]
    return flat


def read_baseline_elemental(checkpoint_dir: str) -> Dict[int, float]:
    """Read a checkpoint's ``model/baseline.yaml`` ``elemental`` offsets.
    Uses plain YAML parsing (no jax dependency) so it works in the torch env
    too. Falls back to :data:`BASELINE_ELEMENTAL_OFFSETS` (this checkpoint's
    already-known values) if PyYAML is unavailable."""
    path = Path(checkpoint_dir) / "model" / "baseline.yaml"
    try:
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh)
        return {int(k): float(v) for k, v in data["elemental"].items()}
    except Exception:
        return dict(BASELINE_ELEMENTAL_OFFSETS)


# --------------------------------------------------------------------------
# weight-loading helpers
# --------------------------------------------------------------------------


def _degree_tag(l: int) -> str:
    return f"{l}{'+' if l % 2 == 0 else '-'}"


def _t(arr) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr))


def _set_linear(linear: torch.nn.Linear, params, prefix: str, bias: bool = True) -> None:
    linear.weight.data.copy_(_t(params[f"{prefix}/kernel"]).T)
    if bias:
        linear.bias.data.copy_(_t(params[f"{prefix}/bias"]))


def _set_layernorm(ln: torch.nn.LayerNorm, params, prefix: str) -> None:
    ln.weight.data.copy_(_t(params[f"{prefix}/scale"]))
    ln.bias.data.copy_(_t(params[f"{prefix}/bias"]))


def _set_update(update, params, prefix: str) -> None:
    """``_JaxUpdate``: mlp0 (Linear, SiLU, Linear), norm0, mlp1, norm1."""
    _set_linear(update.mlp0[0], params, f"{prefix}/MLP_0/Dense_0")
    _set_linear(update.mlp0[2], params, f"{prefix}/MLP_0/Dense_1")
    _set_layernorm(update.norm0, params, f"{prefix}/LayerNorm_0")
    _set_linear(update.mlp1[0], params, f"{prefix}/MLP_1/Dense_0")
    _set_linear(update.mlp1[2], params, f"{prefix}/MLP_1/Dense_1")
    _set_layernorm(update.norm1, params, f"{prefix}/LayerNorm_1")


def _set_energy_mlp(mlp: torch.nn.Sequential, params, prefix: str) -> None:
    _set_linear(mlp[0], params, f"{prefix}/Dense_0")
    _set_linear(mlp[2], params, f"{prefix}/Dense_1")
    _set_linear(mlp[4], params, f"{prefix}/Dense_2")


def _set_degreewise_linear(dwl, params, prefix: str, bias: bool = True) -> None:
    for l, layer in enumerate(dwl.layers):
        tag = _degree_tag(l)
        layer.weight.data.copy_(_t(params[f"{prefix}/{tag}/kernel"]).T)
        if bias and layer.bias is not None:
            key = f"{prefix}/{tag}/bias"
            if key in params:
                layer.bias.data.copy_(_t(params[key]))


def _set_tensor_dense(td, params, prefix: str, use_bias: bool = False) -> None:
    """``TensorDense``: per-degree ``dense`` projection + CG self-product.
    ``tensor/kernel`` is e3x's ``(1, l1, 1, l2, 1, L, features)`` layout
    (``include_pseudotensors=False`` collapses the parity axes to size 1);
    each ``(l1, l2, L)`` coupling's learned per-feature weight is a direct
    slice, no permutation needed (weights don't depend on m-ordering
    convention). The CG buffer itself needs the phase correction, per
    ``e3x_compat.cg_phase_correction`` (see that module's docstring)."""
    _set_degreewise_linear(td.dense, params, f"{prefix}/dense", bias=use_bias)
    kernel = np.asarray(params[f"{prefix}/tensor/kernel"])
    with torch.no_grad():
        for index, coupling in enumerate(td.couplings):
            l1, l2, L = coupling.l1, coupling.l2, coupling.L
            weight = kernel[0, l1, 0, l2, 0, L, :]
            td.tensor_weight[index].copy_(_t(weight))
            if e3x_compat.cg_phase_correction(l1, l2, L) < 0:
                coupling.cg.mul_(-1.0)


def _set_tensor_product(tp, params, prefix: str) -> None:
    kernel = np.asarray(params[f"{prefix}/kernel"])
    with torch.no_grad():
        for index, coupling in enumerate(tp.couplings):
            l1, l2, L = coupling.l1, coupling.l2, coupling.L
            weight = kernel[0, l1, 0, l2, 0, L, :]
            tp.tensor_weight[index].copy_(_t(weight))
            if e3x_compat.cg_phase_correction(l1, l2, L) < 0:
                coupling.cg.mul_(-1.0)


class FixedJaxParityLongRange(JaxParityLongRange):
    """``JaxParityLongRange`` with the two non-PBC-branch bugs fixed (see
    this module's top-level docstring for the full derivation and the
    validation numbers that confirm the fix)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from torchpme import Calculator as _PMECalculator
        from torchpme import CoulombPotential as _CoulombPotential

        # Bug (1): no near-field exclusion on the non-PBC direct sum.
        self.direct_calculator = _PMECalculator(
            potential=_CoulombPotential(smearing=None, exclusion_radius=None),
            full_neighbor_list=False,
        )

    def _potentials(self, systems, charges, neighbor_distances):
        # Bug (2): restore the training-time "no /2" convention.
        return super()._potentials(systems, charges, neighbor_distances) * 2.0


def build_and_load_jax_parity_model(
    params_flat: Dict[str, np.ndarray],
    atomic_types: Sequence[int],
    dtype: torch.dtype = torch.float64,
    fix_lr_bugs: bool = True,
):
    """Build ``JaxParityBackbone`` + (Fixed)``JaxParityLongRange`` and copy
    every weight from ``params_flat`` (see :func:`load_params_flat`) in.

    :param atomic_types: distinct atomic numbers the model will see (unused
        by the modules themselves -- species are embedded directly by
        atomic number -- but required by their constructor signature).
    :param fix_lr_bugs: apply the two non-PBC bug fixes documented on
        :class:`FixedJaxParityLongRange`. Only disable this for debugging /
        reproducing the *unfixed* behavior; leave it on otherwise.
    :return: ``(backbone, long_range, neighbor_list_options)``.
    """
    nl_opts = NeighborListOptions(cutoff=CUTOFF, full_list=True, strict=True)

    backbone = JaxParityBackbone(
        cutoff=CUTOFF,
        max_degree=MAX_DEGREE,
        num_features=NUM_FEATURES,
        num_radial=NUM_RADIAL,
        num_spherical_features=NUM_SPHERICAL_FEATURES,
        num_species=NUM_SPECIES,
        atomic_types=list(atomic_types),
        neighbor_list_options=nl_opts,
    )
    lr_cls = FixedJaxParityLongRange if fix_lr_bugs else JaxParityLongRange
    long_range = lr_cls(
        feature_dim=NUM_FEATURES,
        num_spherical_features=NUM_SPHERICAL_FEATURES,
        max_degree=MAX_DEGREE,
        max_degree_lr=MAX_DEGREE_LR,
        neighbor_list_options=nl_opts,
        smearing=1.0,  # unused: never reached (all-non-PBC test structures)
        kspace_resolution=0.5,  # unused: ditto
    )

    p = params_flat

    backbone.chemical_embedding.weight.data.copy_(
        _t(p["Initial_0/ChemicalEmbedding_0/Embed_0/embedding"])
    )
    _set_linear(backbone.radial_coefficients[0], p, "RadialCoefficients_0/MLP_0/Dense_0")
    _set_linear(backbone.radial_coefficients[2], p, "RadialCoefficients_0/MLP_0/Dense_1")
    _set_linear(backbone.dense0, p, "Dense_0", bias=True)
    _set_linear(backbone.dense1, p, "Dense_1", bias=False)
    _set_update(backbone.update0, p, "Update_0")
    _set_linear(backbone.dense2, p, "Dense_2", bias=False)
    _set_tensor_dense(backbone.tensor_dense, p, "TensorDense_0", use_bias=False)
    _set_update(backbone.update1, p, "Update_1")
    _set_energy_mlp(backbone.energy_mlp, p, "MLP_0")

    _set_linear(long_range.scalar_charge_mlp[0], p, "MLP_1/Dense_0")
    _set_linear(long_range.scalar_charge_mlp[2], p, "MLP_1/Dense_1")
    _set_tensor_dense(long_range.spherical_charge_dense, p, "TensorDense_1", use_bias=False)
    _set_degreewise_linear(long_range.potential_to_features, p, "Dense_3", bias=False)
    _set_tensor_product(long_range.potential_product, p, "Tensor_0")
    _set_update(long_range.update2, p, "Update_2")
    _set_energy_mlp(long_range.energy_mlp, p, "MLP_2")

    backbone = backbone.to(dtype)
    long_range = long_range.to(dtype)
    backbone.bernstein_coeff = backbone.bernstein_coeff.to(dtype)
    backbone.eval()
    long_range.eval()

    return backbone, long_range, nl_opts


# --------------------------------------------------------------------------
# ASE calculator
# --------------------------------------------------------------------------


class JaxParityASECalculator(_ASECalculator):
    """ASE-compatible calculator wrapping the torch-ported lorem-jax SN2
    checkpoint (``JaxParityBackbone`` + ``FixedJaxParityLongRange``), for
    isolated (non-periodic) molecules -- the only case exercised/validated
    here. Adds the elemental baseline offset itself (see this module's
    docstring), matching ``lorem.calculator.Calculator``'s ``add_offset``
    behavior."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CHECKPOINT,
        params_flat: Optional[Dict[str, np.ndarray]] = None,
        params_npz: Optional[str] = None,
        atomic_types: Optional[Sequence[int]] = None,
        baseline: Optional[Dict[int, float]] = None,
        dtype: torch.dtype = torch.float64,
        fix_lr_bugs: bool = True,
        **kwargs,
    ) -> None:
        """Build the calculator from checkpoint weights.

        This class lives in the torch environment (imports
        ``metatomic.torch`` etc.), which does not have ``lorem``/``jax``
        installed -- so in practice, weights should be supplied via
        ``params_npz`` (a flat ``.npz`` produced once, in the *jax* venv, by
        :func:`dump_checkpoint_params_to_npz`) rather than ``checkpoint_dir``
        directly. ``checkpoint_dir`` alone (``params_flat=None``,
        ``params_npz=None``) only works if this happens to run in an
        environment with ``lorem``/``jax`` importable.
        """
        super().__init__(**kwargs)
        if params_flat is None:
            if params_npz is not None:
                params_flat = load_params_flat(params_npz)
            else:
                params_flat = read_checkpoint_params(checkpoint_dir)
        if atomic_types is None:
            atomic_types = [1, 6, 7, 8, 9, 17, 35, 53]  # any species seen; unused internally
        if baseline is None:
            baseline = read_baseline_elemental(checkpoint_dir)

        self.dtype = dtype
        self.baseline = baseline
        self.backbone, self.long_range, self.nl_opts = build_and_load_jax_parity_model(
            params_flat, atomic_types, dtype=dtype, fix_lr_bugs=fix_lr_bugs
        )

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        positions = torch.tensor(
            atoms.get_positions(), dtype=self.dtype, requires_grad=True
        )
        numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32)
        cell = torch.zeros((3, 3), dtype=self.dtype)
        pbc = torch.tensor([False, False, False])

        system = System(types=numbers, positions=positions, cell=cell, pbc=pbc)
        system = get_system_with_neighbor_lists(system, [self.nl_opts])

        nodes_scalar, distances, nodes_spherical, sr_energy = self.backbone([system])
        lr_energy = self.long_range([system], nodes_scalar, distances, nodes_spherical)
        per_atom_energy = sr_energy + lr_energy

        offsets = torch.tensor(
            [self.baseline[int(z)] for z in atoms.get_atomic_numbers()], dtype=self.dtype
        )
        total_energy = per_atom_energy.sum() + offsets.sum()

        (grad,) = torch.autograd.grad(total_energy, positions)
        forces = -grad.detach().numpy()

        self.results["energy"] = float(total_energy.detach().numpy())
        self.results["forces"] = forces
