"""PETLR with a switchable Ewald backend, plus the matching predict path."""

import jax
import jax.numpy as jnp

import flax.linen as nn
from marathon.utils import masked
from petjax import MLP
from petjax.select import pack_edges, truncate_edges

from iris.pet.batching import SR, to_jaxpme
from iris.pet.model import PET
from iris.shared.megabatch import component_arrays

from mixed_lr import SHARED, to_jaxpme_mixed


def ewald(backend, prefactor):
    if backend == "tiled":
        from jaxpme.batched_tiled import Ewald

        return Ewald(prefactor=prefactor)
    from jaxpme.batched_mixed import Ewald

    return Ewald(prefactor=prefactor)


def lr_adapter(backend):
    return to_jaxpme if backend == "tiled" else to_jaxpme_mixed


class PETLRBench(nn.Module):
    backend: str = "tiled"

    d_pet: int = 128
    d_node: int = 128
    d_head: int = 128
    d_feedforward: int = 256
    num_heads: int = 8
    num_attention_layers: int = 1
    num_gnn_layers: int = 2
    cutoff: float = 5.0
    cutoff_width: float = 0.5
    cutoff_width_adaptive: float = 0.5
    adaptive_cutoff_method: str = "grid"
    num_neighbors_adaptive: int = 8
    attention_temperature: float = 1.0
    max_atomic_number: int = 118
    direct_forces: bool = False
    direct_stress: bool = False

    fixed_neighbors: int | None = None
    no_shadow: bool = False

    lr: bool = True
    num_charges: int = 8
    d_lr: int | None = None
    init_prefactor: float = 1.0
    lr_scale_init: float = 1.0

    @nn.compact
    def __call__(self, truncated, lr):
        out, node = PET(
            d_pet=self.d_pet,
            d_node=self.d_node,
            d_head=self.d_head,
            d_feedforward=self.d_feedforward,
            num_heads=self.num_heads,
            num_attention_layers=self.num_attention_layers,
            num_gnn_layers=self.num_gnn_layers,
            cutoff=self.cutoff,
            cutoff_width=self.cutoff_width,
            num_neighbors_adaptive=self.num_neighbors_adaptive,
            attention_temperature=self.attention_temperature,
            max_atomic_number=self.max_atomic_number,
            direct_forces=self.direct_forces,
            direct_stress=self.direct_stress,
            name="sr",
        )(**truncated, return_features=True)

        if not self.lr:
            return out

        lr_sr, lr_nopbc, lr_pbc = lr
        lr_energy = _LRModule(
            backend=self.backend,
            d_node=self.d_node,
            d_feedforward=self.d_feedforward,
            num_charges=self.num_charges,
            d_lr=self.d_lr,
            init_prefactor=self.init_prefactor,
            lr_scale_init=self.lr_scale_init,
            name="lr",
        )(node, truncated["atom_mask"], lr_sr, lr_nopbc, lr_pbc)

        if isinstance(out, dict):
            return {**out, "energy": out["energy"] + lr_energy}
        return out + lr_energy


class _LRModule(nn.Module):
    backend: str = "tiled"
    d_node: int = 128
    d_feedforward: int = 256
    num_charges: int = 8
    d_lr: int | None = None
    init_prefactor: float = 1.0
    lr_scale_init: float = 1.0

    @nn.compact
    def __call__(self, node_embedding, atom_mask, sr_batch, nopbc, pbc):
        d_hidden = self.d_node if self.d_lr is None else self.d_lr
        d_mix = self.d_feedforward if self.d_lr is None else self.d_lr

        charges = masked(
            MLP((d_hidden, d_hidden, self.num_charges)),
            node_embedding,
            atom_mask,
        )

        prefactor = jnp.exp(
            self.param(
                "log_prefactor",
                nn.initializers.constant(jnp.log(self.init_prefactor)),
                (),
            )
        )
        calculator = ewald(self.backend, prefactor)

        potentials = jax.vmap(
            lambda q: calculator.potentials(q, sr_batch, nopbc, pbc),
            in_axes=-1,
            out_axes=-1,
        )(charges)

        h = node_embedding + masked(MLP((d_mix, self.d_node)), potentials, atom_mask)
        h = masked(nn.LayerNorm(), h, atom_mask)
        h = h + masked(MLP((d_mix, self.d_node)), h, atom_mask)
        h = masked(nn.LayerNorm(), h, atom_mask)

        lr_energy = masked(MLP((d_hidden, d_hidden, 1)), h, atom_mask)[:, 0]

        lr_scale = self.param(
            "lr_scale", nn.initializers.constant(self.lr_scale_init), (1,)
        )
        return lr_energy * lr_scale


# -- predict (iris.pet.predict, with the LR adapter injected) -------------------


def _displacements(batch):
    return (
        batch["positions"][batch["others"]]
        - batch["positions"][batch["centers"]]
        + jnp.einsum(
            "pa,pab->pb", batch["cell_shifts"], batch["cell"][batch["pair_to_structure"]]
        )
    )


def model_inputs(model, batch):
    sr = component_arrays(batch, "sr", SR().shared_arrays)
    R_ij = _displacements(sr)
    width = sr["k_sel_sizer"].shape[-1]
    args = (
        R_ij,
        sr["centers"],
        sr["others"],
        sr["reverse"],
        sr["pair_mask"],
        sr["atomic_numbers"],
        sr["atom_mask"],
    )
    if model.fixed_neighbors is None:
        truncated, overflow = truncate_edges(
            *args,
            width,
            model.num_neighbors_adaptive,
            model.cutoff,
            model.cutoff_width_adaptive,
            method=model.adaptive_cutoff_method,
            no_shadow=model.no_shadow,
        )
    else:
        truncated, overflow = pack_edges(*args, width)

    lr = None
    if getattr(model, "lr", False):
        lr = lr_adapter(model.backend)(component_arrays(batch, "lr", SHARED))
    return truncated, lr, overflow


def energy(model, params, batch):
    truncated, lr, overflow = model_inputs(model, batch)
    out = model.apply(params, truncated, lr)
    per_atom = out["energy"] if isinstance(out, dict) else out
    return jnp.sum(per_atom), (per_atom, overflow)


def predict(model, params, batch, stress=True):
    def energy_fn(positions, cell):
        return energy(model, params, {**batch, "positions": positions, "cell": cell})

    (_, (per_atom, overflow)), (g_pos, g_cell) = jax.value_and_grad(
        energy_fn, argnums=(0, 1), has_aux=True
    )(batch["positions"], batch["cell"])

    S = batch["structure_mask"].shape[0]
    results = {
        "energy": jax.ops.segment_sum(per_atom, batch["atom_to_structure"], S),
        "forces": -g_pos * batch["atom_mask"][:, None],
        "overflow": overflow,
    }
    if stress:
        virial = jax.ops.segment_sum(
            jnp.einsum("ia,ib->iab", batch["positions"], g_pos),
            batch["atom_to_structure"],
            S,
        ) + jnp.einsum("sAa,sAb->sab", batch["cell"], g_cell)
        results["stress"] = virial * batch["structure_mask"][:, None, None]
    return results
