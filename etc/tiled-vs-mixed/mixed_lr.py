"""batched_mixed counterparts of iris.pet.batching.TiledEwaldLR / to_jaxpme."""

import numpy as np

from dataclasses import dataclass

from iris.shared.megabatch import Axis

SHARED = ("positions", "cell", "atom_to_structure", "structure_mask", "atom_mask")


@dataclass(frozen=True)
class MixedEwaldLR:
    @property
    def spec(self):
        return {
            "n_structures": Axis("sum", 1),
            "n_atoms": Axis("sum", 1),
            "n_pairs": Axis("sum", 1),
            "n_structures_pbc": Axis("sum", 1),
            "n_atoms_pbc": Axis("max", 0),
            "n_pairs_nonpbc": Axis("sum", 1),
            "num_k": Axis("max", 0),
        }

    @property
    def discretizers(self):
        return {
            "n_structures": None,
            "n_atoms": "multiples",
            "n_pairs": "multiples",
            "n_structures_pbc": None,
            "n_atoms_pbc": None,
            "n_pairs_nonpbc": "multiples",
            "num_k": None,
        }

    @property
    def shared_arrays(self):
        return SHARED

    def sizer(self, sample):
        s = sample.lr_structure
        lr = s["lr"]
        is_pbc = hasattr(lr, "k_grid")
        n = len(s["positions"])
        return {
            "n_structures": 1,
            "n_atoms": n,
            "n_pairs": len(s["centers"]),
            "n_structures_pbc": int(is_pbc),
            "n_atoms_pbc": n if is_pbc else 0,
            "n_pairs_nonpbc": 0 if is_pbc else len(lr.centers),
            "num_k": int(lr.k_grid.shape[0]) if is_pbc else 0,
        }

    def materialize(self, samples, shapes):
        from jaxpme.batched_mixed.batching import get_batch

        return _flatten_mixed(
            get_batch(
                [s.lr_structure for s in samples],
                num_structures=shapes["n_structures"],
                num_structures_pbc=shapes["n_structures_pbc"],
                num_atoms=shapes["n_atoms"],
                num_atoms_pbc=shapes["n_atoms_pbc"],
                num_pairs=shapes["n_pairs"],
                num_pairs_nonpbc=shapes["n_pairs_nonpbc"],
                num_k=shapes["num_k"],
            )
        )

    def info(self, view):
        real = {
            "pairs": view["pair_mask"].sum(),
            "atoms_pbc": view["pbc_atom_mask"].sum(),
            "pbc": view["pbc_structure_mask"].sum(),
            "pairs_nonpbc": view["nonpbc_pair_mask"].sum(),
            "k": (view["k_grid"] != 0).any(axis=-1).sum(),
        }
        total = {
            "pairs": view["pair_mask"].size,
            "atoms_pbc": view["pbc_atom_mask"].size,
            "pbc": view["pbc_structure_mask"].size,
            "pairs_nonpbc": view["nonpbc_pair_mask"].size,
            "k": view["k_grid"].shape[-3] * view["k_grid"].shape[-2],
        }
        shape = {
            "pairs": view["pair_mask"].shape[-1],
            "atoms_pbc": view["pbc_atom_mask"].shape[-1],
            "pbc": view["pbc_structure_mask"].shape[-1],
            "pairs_nonpbc": view["nonpbc_pair_mask"].shape[-1],
            "k": view["k_grid"].shape[-2],
        }
        return real, total, shape


def _f32(x):
    return np.asarray(x, dtype=np.float32)


def _i32(x):
    return np.asarray(x, dtype=np.int32)


def _flatten_mixed(batch):
    _charges, sr, nopbc, pbc = batch
    return {
        "positions": _f32(sr.positions),
        "cell": _f32(sr.cell),
        "atom_to_structure": _i32(sr.atom_to_structure),
        "structure_mask": sr.structure_mask,
        "atom_mask": sr.atom_mask,
        "effective_cell": _f32(sr.effective_cell),
        "pbc_rows": sr.pbc,
        "smearing": _f32(sr.smearing),
        "centers": _i32(sr.centers),
        "others": _i32(sr.others),
        "cell_shifts": _i32(sr.cell_shifts),
        "pair_mask": sr.pair_mask,
        "pair_to_structure": _i32(sr.pair_to_structure),
        "pbc_mask": sr.pbc_mask,
        "nonpbc_centers": _i32(nopbc.centers),
        "nonpbc_others": _i32(nopbc.others),
        "nonpbc_pair_mask": nopbc.pair_mask,
        "k_grid": _f32(pbc.k_grid),
        "pbc_atom_to_atom": _i32(pbc.atom_to_atom),
        "pbc_structure_to_structure": _i32(pbc.structure_to_structure),
        "pbc_atom_mask": pbc.atom_mask,
        "pbc_structure_mask": pbc.structure_mask,
        "pbc": pbc.pbc,
    }


def to_jaxpme_mixed(view):
    from jaxpme.batched_mixed.batching import Batch, NonPeriodic, Periodic

    sr = Batch(
        positions=view["positions"],
        cell=view["cell"],
        effective_cell=view["effective_cell"],
        pbc=view["pbc_rows"],
        smearing=view["smearing"],
        centers=view["centers"],
        others=view["others"],
        cell_shifts=view["cell_shifts"],
        atom_mask=view["atom_mask"],
        pair_mask=view["pair_mask"],
        structure_mask=view["structure_mask"],
        pbc_mask=view["pbc_mask"],
        atom_to_structure=view["atom_to_structure"],
        pair_to_structure=view["pair_to_structure"],
        distances=None,
    )
    nonperiodic = NonPeriodic(
        centers=view["nonpbc_centers"],
        others=view["nonpbc_others"],
        pair_mask=view["nonpbc_pair_mask"],
    )
    periodic = Periodic(
        k_grid=view["k_grid"],
        atom_to_atom=view["pbc_atom_to_atom"],
        structure_to_structure=view["pbc_structure_to_structure"],
        atom_mask=view["pbc_atom_mask"],
        structure_mask=view["pbc_structure_mask"],
        pbc=view["pbc"],
    )
    return sr, nonperiodic, periodic
