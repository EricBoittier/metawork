"""
Registry of SN2 long-range-scan models used by the `0{3..11}_long_range_scan_*`
notebooks and `12_long_range_summary.ipynb`.

Each entry describes one trained `model.pt` under `../data/` that is loadable
with `MetatomicCalculator` and evaluable on the shared `sn2/sn2.xyz` reference
structure. Directories that only contain an `sn2.xyz` + empty `outputs/` (no
`model.pt` -- i.e. a training run that never produced/exported a model, such
as `sn2-bpnn`, `sn2-bpnn-dipole`, `sn2-dipole`, `sn2-pet`, `sn2-pet-dipole`,
`diag-bpnn-w20`) are not included here; there is nothing to load.
"""

from pathlib import Path

DATA = Path("/home/boittier/data")
RESULTS = Path(__file__).parent / "results"

MODELS = [
    {
        "key": "sn2-matched-bpnn",
        "notebook": "02_long_range_scan.ipynb",
        "title": "SN2 -- matched BPNN",
        "family": "BPNN",
        "dipole": False,
        "dir": DATA / "sn2-matched-bpnn",
        "note": "Baseline BPNN model, matched training set. Covered by "
                "02_long_range_scan.ipynb.",
    },
    {
        "key": "sn2",
        "notebook": "03_long_range_scan_sn2.ipynb",
        "title": "SN2 -- original (unmatched) model",
        "family": "BPNN",
        "dipole": False,
        "dir": DATA / "sn2",
        "note": "The original model from 01_load_models.ipynb, trained "
                "before the matched-dataset re-runs; also ships a "
                "diag_model.ckpt training checkpoint alongside model.ckpt.",
    },
    {
        "key": "sn2-matched-bpnn-dipole",
        "notebook": "04_long_range_scan_bpnn_dipole.ipynb",
        "title": "SN2 -- matched BPNN + dipole",
        "family": "BPNN",
        "dipole": True,
        "dir": DATA / "sn2-matched-bpnn-dipole",
        "note": "BPNN with an added dipole/long-range electrostatic term.",
    },
    {
        "key": "sn2-matched-bpnn-dipole-v2",
        "notebook": "05_long_range_scan_bpnn_dipole_v2.ipynb",
        "title": "SN2 -- matched BPNN + dipole (v2)",
        "family": "BPNN",
        "dipole": True,
        "dir": DATA / "sn2-matched-bpnn-dipole-v2",
        "note": "Second iteration of the BPNN+dipole model.",
    },
    {
        "key": "sn2-matched-bpnn-dipole-v3",
        "notebook": "06_long_range_scan_bpnn_dipole_v3.ipynb",
        "title": "SN2 -- matched BPNN + dipole (v3)",
        "family": "BPNN",
        "dipole": True,
        "dir": DATA / "sn2-matched-bpnn-dipole-v3",
        "note": "Third iteration of the BPNN+dipole model.",
    },
    {
        "key": "sn2-matched-lorem",
        "notebook": "07_long_range_scan_lorem.ipynb",
        "title": "SN2 -- matched LOREM",
        "family": "LOREM",
        "dipole": False,
        "dir": DATA / "sn2-matched-lorem",
        "note": "LOREM (long-range equivariant message passing) architecture, "
                "https://arxiv.org/abs/2507.19382.",
    },
    {
        "key": "sn2-matched-lorem-dipole",
        "notebook": "08_long_range_scan_lorem_dipole.ipynb",
        "title": "SN2 -- matched LOREM + dipole",
        "family": "LOREM",
        "dipole": True,
        "dir": DATA / "sn2-matched-lorem-dipole",
        "note": "LOREM with an added dipole/long-range electrostatic term.",
    },
    {
        "key": "sn2-matched-pet",
        "notebook": "09_long_range_scan_pet.ipynb",
        "title": "SN2 -- matched PET",
        "family": "PET",
        "dipole": False,
        "dir": DATA / "sn2-matched-pet",
        "note": "PET (point-edge transformer) architecture, "
                "https://arxiv.org/abs/2305.19302.",
    },
    {
        "key": "sn2-matched-pet-dipole",
        "notebook": "10_long_range_scan_pet_dipole.ipynb",
        "title": "SN2 -- matched PET + dipole",
        "family": "PET",
        "dipole": True,
        "dir": DATA / "sn2-matched-pet-dipole",
        "note": "PET with an added dipole/long-range electrostatic term.",
    },
    {
        "key": "lorem-eqmp-smoketest",
        "notebook": "11_long_range_scan_lorem_eqmp_smoketest.ipynb",
        "title": "LOREM EQMP smoketest",
        "family": "LOREM",
        "dipole": False,
        "dir": DATA / "lorem-eqmp-smoketest",
        "note": "Not part of the sn2-matched-* series -- a separate smoketest "
                "run of an equivariant message-passing LOREM variant "
                "(see diag.yaml in this directory), included here as 'the "
                "other model' with a usable model.pt.",
    },
]

# --- native lorem-jax checkpoints (not metatrain/torch) ---------------------
#
# These are NOT trained on the same data as the sn2-matched-* models above --
# they are the official lorem-tmlr-archive reproduction checkpoints for a
# broader SN2 halide-exchange benchmark (X- + CH3-Y, X,Y in {F,Cl,Br,I}), so
# their absolute energies are not directly comparable to the sn2-matched-*
# family; only the qualitative long-range-cutoff behavior is. They require a
# different interpreter: /home/boittier/metawork/.venv-lorem-jax/bin/python
# (has `lorem`/jax installed; the regular metawork/.venv does not), and a
# different calculator API: lorem.calculator.Calculator.from_checkpoint(...)
# (ASE-compatible) instead of MetatomicCalculator.
LOREM_JAX_VENV_PYTHON = "/home/boittier/metawork/.venv-lorem-jax/bin/python"
LOREM_TMLR_ARCHIVE = Path("/home/boittier/metawork/lorem-tmlr-archive")

LOREM_JAX_MODELS = [
    {
        "key": "lorem-jax-sn2-lr",
        "notebook": "13_long_range_scan_lorem_jax_lr.ipynb",
        "title": "SN2 (jax) -- lorem-jax, long-range (Ewald)",
        "family": "LOREM-jax",
        "dipole": False,
        "checkpoint": LOREM_TMLR_ARCHIVE / "evals/sn2/lorem/run/checkpoints/R2_E+F",
        "lr_enabled": True,
        "note": (
            "Official lorem-tmlr-archive checkpoint (lr: true, max_degree: 6, "
            "num_features: 128), loaded via lorem.calculator.Calculator. "
            "The shipped non-PBC LOREM checkpoints were trained before a "
            "jax-pme convention change that halves the non-PBC long-range "
            "potential (see lorem-tmlr-archive/README.md, 'Non-PBC LR Coulomb "
            "convention'); this notebook applies the documented x2 "
            "correction (an in-process patch of the Ewald() potentials call "
            "in lorem.models.mlip, not a change to any installed file) and "
            "validates it against the archive's own frozen "
            "minimum_energy_path.npz predictions before using it on the "
            "sn2/sn2.xyz reference structure."
        ),
    },
    {
        "key": "lorem-jax-sn2-sr",
        "notebook": "14_long_range_scan_lorem_jax_sr.ipynb",
        "title": "SN2 (jax) -- lorem-jax, short-range only",
        "family": "LOREM-jax",
        "dipole": False,
        "checkpoint": LOREM_TMLR_ARCHIVE / "evals/sn2/lorem-sr-mp2/run/checkpoints/R2_E+F",
        "lr_enabled": False,
        "note": (
            "Official lorem-tmlr-archive checkpoint (lr: false, "
            "equivariant_message_passing: true, num_message_passing: 1), "
            "loaded via lorem.calculator.Calculator. Does not invoke the "
            "Ewald/Coulomb path at all, so it is unaffected by the non-PBC "
            "convention change described above -- usable as-is."
        ),
    },
]

REFERENCE_XYZ = DATA / "sn2" / "sn2.xyz"  # shared reaction-coordinate geometry

MODELS_BY_KEY = {m["key"]: m for m in MODELS}
LOREM_JAX_MODELS_BY_KEY = {m["key"]: m for m in LOREM_JAX_MODELS}

# --- torch port of the real lorem-jax checkpoint (JaxParityBackbone/JaxParityLongRange) ---
#
# Unlike LOREM_JAX_MODELS above (native lorem-jax, needs the jax venv), this
# one runs in the SAME torch environment as MODELS (metawork/.venv) via a
# from-scratch weight-loading port -- see 15_long_range_scan_lorem_jax_ported_torch.ipynb
# and lorem_jax_ported_torch.py for the full validation (matches the live
# jax reference to ~2e-5 eV / ~3e-5 eV/Å) and the two non-PBC long-range bugs
# found and fixed along the way.
PORTED_TORCH_MODELS = [
    {
        "key": "lorem-jax-sn2-lr-ported-torch",
        "notebook": "15_long_range_scan_lorem_jax_ported_torch.ipynb",
        "title": "SN2 (jax-ported torch) -- lorem-jax checkpoint via JaxParityBackbone/JaxParityLongRange",
        "family": "LOREM-jax",
        "dipole": False,
        "note": (
            "Same lorem-jax R2_E+F checkpoint as lorem-jax-sn2-lr, loaded "
            "here via a from-scratch torch port (JaxParityBackbone + "
            "JaxParityLongRange, metatrain/.../lorem/modules/jax_parity.py) "
            "instead of the native jax lorem.calculator.Calculator. "
            "Validated against the live jax reference to ~2e-5 eV / "
            "~3e-5 eV/Å (see notebook 15) -- essentially float32-noise-"
            "level agreement, after fixing two bugs in the non-PBC "
            "long-range branch (a wrongly-inherited near-field exclusion "
            "radius, and a missing x2 training/production convention "
            "factor -- see lorem_jax_ported_torch.py's module docstring)."
        ),
    },
]

PORTED_TORCH_MODELS_BY_KEY = {m["key"]: m for m in PORTED_TORCH_MODELS}
