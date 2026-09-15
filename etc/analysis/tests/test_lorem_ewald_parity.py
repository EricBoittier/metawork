"""Regression tests guarding the class of bug documented in notebooks 13/14 (lorem-jax's
non-PBC vs. PBC Ewald convention), 16 (`exclusion_radius` wiring in a periodic Ewald
torch port), 17 (charged-structure NVE engine parity), and 20 (neutral-structure NVE
engine parity) -- LOREM's long-range/Ewald term has repeatedly been the place where a
periodicity assumption or a checkpoint/architecture mismatch silently breaks parity
between two things that are supposed to agree (two engines, or a checkpoint and the
code that loads it).

Two independent failure classes are covered here:

1. **Checkpoint/architecture drift** (`test_v3_checkpoints_are_rejected_loudly`):
   `metatrain.experimental.lorem.model.LOREM.load_checkpoint` used to call
   `load_state_dict(..., strict=False)`, which *silently* left any missing parameter
   at its random initialization instead of raising -- so a checkpoint saved before an
   architecture refactor (e.g. the equivariant-message-passing / TensorDense-CG
   refactors, commits `700c0b73`/`ce58eaae`) reloaded "successfully" into a model that
   was part-trained, part-random, with no warning and non-reproducible outputs run to
   run. **This has been fixed upstream** (metatrain, `experimental/lorem` branch):
   `load_state_dict` is strict again, and `LOREM.__checkpoint_version__` was bumped to
   4 with an explicit, unconditional `model_update_v3_v4` that refuses to upgrade any
   version-3 checkpoint (no state-dict migration from the old `TensorDense`
   `proj_a`/`proj_b` parameterization to the new CG-coupling one is mathematically
   possible -- it's new capacity, not a rename). Every local LOREM checkpoint still
   stamped version 3 -- `sn2-matched-lorem`, `sn2-matched-lorem-dipole`, and even
   `lorem-eqmp-smoketest` (whose *state dict* already happened to match the current
   architecture, but whose stamped version number was never bumped when that
   architecture landed) -- is now rejected the same way: loudly, with a clear
   `RuntimeError` naming the refactor and telling the caller to retrain. Only the
   already-*exported* `model.pt` for these checkpoints remains usable, for inference
   only; a fresh, version-4 `sn2-matched-lorem` checkpoint is being retrained from the
   original recipe (`metawork/etc/sn2_zenodo/options/matched-budget/
   energy-forces-lorem.yaml`) to replace it -- see `sn2-matched-lorem-v4/` once that
   finishes (a multi-hour GPU job).

2. **Cross-engine NVE energy parity** (`test_*_engine_parity` /
   `test_lorem_periodic_engines_agree_with_each_other`): reuses the already-validated
   `hourglass_engines.py` runners (ASE, i-PI, LAMMPS) on the exported, deterministic
   `model.pt` artifacts -- the same artifacts notebooks 17/20 report on -- to pin down,
   as an executable regression test, what "parity" and "expected non-parity" look like
   for this model family:
     - PET (short-range only) must agree across ASE/i-PI/LAMMPS and across box sizes,
       for both a charged and a neutral test structure -- if this regresses, something
       broke in the shared engine plumbing itself, not in LOREM's long-range term.
     - LOREM's two independent *periodic* engines (i-PI, LAMMPS) must keep agreeing
       with *each other* at a fixed box size, on both structures -- this is the
       invariant an `exclusion_radius`-class engine-specific wiring bug would break,
       independent of whether periodic and non-periodic evaluations agree with each
       other (they are not expected to, for this model family -- see notebook 20).

Slow tests (spawn `lmp`/`gmx` subprocesses, or run ~100-step MD) are marked
``@pytest.mark.slow``; run with ``-m "not slow"`` for a quick pass.

Run with the standard torch env from this directory:
    /home/boittier/metawork/.venv/bin/python -m pytest tests/test_lorem_ewald_parity.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import ase.io  # noqa: E402

from hourglass_engines import run_ase, run_ipi, run_lammps  # noqa: E402
from model_registry import MODELS_BY_KEY  # noqa: E402

ANALYSIS_DIR = Path(__file__).parent.parent
LAMMPS_AVAILABLE = Path("/home/boittier/metawork/lammps/build/lmp").exists()

CHARGED_XYZ = ANALYSIS_DIR / "sn2_frame0.xyz"  # 6-atom CH3F...I- complex, net -1
NEUTRAL_XYZ = ANALYSIS_DIR / "ch3f_neutral.xyz"  # 5-atom CH3F, net 0

LOREM_MODEL = str(MODELS_BY_KEY["sn2-matched-lorem"]["dir"] / "model.pt")
PET_MODEL = str(MODELS_BY_KEY["sn2-matched-pet"]["dir"] / "model.pt")

PET_TOLERANCE_MEV = 1.0  # upstream cookbook's own bound; PET meets it with room to spare
LOREM_ENGINE_AGREEMENT_MEV = 1.0  # i-PI vs. LAMMPS, same box -- both periodic, must match


# --- 1. checkpoint/architecture drift -------------------------------------------


def _load_checkpoint_dict(checkpoint_path: str):
    import metatomic.torch  # noqa: F401  (registers ModelMetadata for torch.load)
    import sphericart.torch  # noqa: F401
    import torch

    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


@pytest.mark.parametrize(
    "model_key", ["sn2-matched-lorem", "sn2-matched-lorem-dipole", "lorem-eqmp-smoketest"]
)
def test_v3_checkpoints_are_rejected_loudly(model_key):
    """Every local LOREM checkpoint is currently stamped version 3 -- including
    `lorem-eqmp-smoketest`, whose state dict already matches the current architecture
    but whose version number was never bumped when that architecture landed. Since
    `LOREM.__checkpoint_version__` is now 4 and `model_update_v3_v4` unconditionally
    refuses to upgrade (no state-dict migration is possible), all three must now fail
    loudly and identically -- not silently succeed with random-initialized parameters
    (the bug this test used to document as an `xfail`)."""
    from metatrain.experimental.lorem.model import LOREM

    checkpoint_path = str(MODELS_BY_KEY[model_key]["dir"] / "model.ckpt")
    ckpt = _load_checkpoint_dict(checkpoint_path)
    with pytest.raises(RuntimeError, match="cannot be automatically upgraded"):
        LOREM.upgrade_checkpoint(ckpt)


# --- 2. cross-engine NVE energy parity -------------------------------------------


def _max_abs_delta_meV(times_a, energies_a, times_b, energies_b):
    import numpy as np

    ta, ea = np.array(times_a), np.array(energies_a)
    tb, eb = np.array(times_b), np.array(energies_b)
    assert len(ta) == len(tb) and np.abs(ta - tb).max() < 1e-6, "time grids differ"
    return 1000 * float(np.abs(ea - eb).max())


@pytest.mark.parametrize("structure_xyz", [CHARGED_XYZ, NEUTRAL_XYZ], ids=["charged", "neutral"])
@pytest.mark.parametrize("box", [30.0, 100.0])
def test_pet_ase_ipi_engine_parity(structure_xyz, box):
    """PET is short-range: ASE (non-periodic) and i-PI (periodic) must agree to well
    under 1 meV regardless of box size or the structure's net charge."""
    atoms = ase.io.read(structure_xyz)
    _, ase_e = run_ase(PET_MODEL, atoms, ensemble="nve")
    ase_t = [i * 0.5 for i in range(len(ase_e))]
    ipi_t, ipi_e = run_ipi(PET_MODEL, atoms, ensemble="nve", cell=box, template_xyz=str(structure_xyz))
    delta = _max_abs_delta_meV(ase_t, ase_e, ipi_t, ipi_e)
    assert delta < PET_TOLERANCE_MEV, f"PET ASE/i-PI mismatch at box={box:g} Å: {delta:.4f} meV"


@pytest.mark.slow
@pytest.mark.skipif(not LAMMPS_AVAILABLE, reason="LAMMPS ML-METATOMIC build not found")
@pytest.mark.parametrize("structure_xyz", [CHARGED_XYZ, NEUTRAL_XYZ], ids=["charged", "neutral"])
def test_lorem_periodic_engines_agree_with_each_other(structure_xyz):
    """i-PI and LAMMPS both treat their cell as periodic and both implement the same
    Ewald long-range term LOREM predicts charges for -- at a fixed box, they must agree
    with *each other* to sub-meV, regardless of whether either agrees with the
    non-periodic ASE reference (they don't, for this model family -- see notebook 20;
    that is a separate, expected physics/model-behavior difference, not an engine bug).
    A mismatch here means one specific periodic engine's Ewald/exclusion-radius wiring
    broke -- exactly the class of bug notebook 16 found in the jax-parity torch port.
    """
    atoms = ase.io.read(structure_xyz)
    box = 50.0
    ipi_t, ipi_e = run_ipi(LOREM_MODEL, atoms, ensemble="nve", cell=box, template_xyz=str(structure_xyz))
    lammps_t, lammps_e = run_lammps(LOREM_MODEL, atoms, ensemble="nve", cell=box)
    delta = _max_abs_delta_meV(ipi_t, ipi_e, lammps_t, lammps_e)
    assert delta < LOREM_ENGINE_AGREEMENT_MEV, (
        f"i-PI/LAMMPS disagree on LOREM at box={box:g} Å: {delta:.4f} meV "
        f"(structure: {structure_xyz.name})"
    )


@pytest.mark.slow
def test_lorem_ase_ipi_disagreement_shrinks_monotonically_with_box():
    """Not a parity assertion (LOREM/ASE are known to disagree here -- see notebook
    20) -- a sanity check that the *direction* of the box-size dependence stays
    physically sane (larger vacuum cell -> smaller periodic-vs-non-periodic gap) for
    both the charged and the neutral structure, guarding against a future regression
    that flips or breaks this trend entirely."""
    import numpy as np

    for structure_xyz in [CHARGED_XYZ, NEUTRAL_XYZ]:
        atoms = ase.io.read(structure_xyz)
        _, ase_e = run_ase(LOREM_MODEL, atoms, ensemble="nve")
        ase_t = [i * 0.5 for i in range(len(ase_e))]
        deltas = []
        for box in [20.0, 50.0, 100.0]:
            ipi_t, ipi_e = run_ipi(
                LOREM_MODEL, atoms, ensemble="nve", cell=box, template_xyz=str(structure_xyz)
            )
            deltas.append(_max_abs_delta_meV(ase_t, ase_e, ipi_t, ipi_e))
        assert deltas == sorted(deltas, reverse=True), (
            f"{structure_xyz.name}: max|ΔU| vs. box size is not monotonically "
            f"decreasing: {deltas}"
        )
