# analysis

A numbered Jupyter notebook series (`01`-`19`) built up across several Claude Code sessions,
mostly tracking the long-range (Coulomb/Ewald) behavior of SN2 halide-exchange models --
LOREM in particular -- and validating the torch port of LOREM against the reference
`lorem-jax` implementation.

Start at **`site/index.html`** (open it directly in a browser, e.g.
`file:///home/boittier/metawork/etc/analysis/site/index.html`) for a browsable landing page
linking every notebook's HTML render and a PDF download. Each notebook is also directly
readable/re-runnable as a plain `.ipynb`.

## Layout

- `01`-`12` -- SN2 reaction-coordinate long-range scans across models (BPNN, LOREM, PET,
  with/without dipole head), cataloged in `model_registry.py` and tested with shared helpers
  in `long_range_tests.py`.
- `13`-`14` -- the official `lorem-tmlr-archive` checkpoint run natively in JAX, as ground
  truth for the torch port.
- `15` -- first torch port of a real lorem-jax checkpoint (non-periodic SN2 case), matching
  to ~2e-5 eV.
- `16` -- the periodic (Ewald, AuMgO/bio_dimers/cumulene) checkpoint-parity investigation:
  finds and fixes a real bug (`JaxParityLongRange` setting a nonzero `exclusion_radius` where
  it should be `None`), documented with the full before/after investigation.
- `17` -- NVE engine parity (ASE/i-PI/...) for LOREM and PET, from the upstream
  `metatomic-hourglass` cookbook recipe.
- `18` -- follow-up to `16`: restarts cumulene and bio_dimers training (with tuned, lower
  learning rates) to see whether ordinary fine-tuning recovers the paper's own reported
  test-set accuracy post-fix. Cumulene matches/beats the paper; bio_dimers improves
  substantially but falls short. Also documents a real methodology trap: a fixed, non-random
  monitoring subset was misleading in opposite directions for the two systems.
- `19` -- a genuine liquid-density ethanol MD run (PET-MAD, via LAMMPS's `pair_style
  metatomic`), from a different peer session's work, pulled in and documented here rather
  than left in an ephemeral session scratchpad.

Supporting code/data: `model_registry.py`, `long_range_tests.py`, `hourglass_engines.py` (ASE/
i-PI/LAMMPS/GROMACS runners), and per-notebook result caches under `results/`,
`metawork54_results/`, `ethanol_liquid_pet_mad_results/`, and `water_metatensor_dataset/`.

## Rebuilding the site

```
cd etc/analysis
./build_site.sh
```

This re-renders every `*.ipynb` to `site/*.html` and `pdf/*.pdf` (via `jupyter nbconvert`,
using the repo's own `.venv`). It does **not** re-execute any notebook -- to refresh a
notebook's own outputs first:

```
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 <notebook>.ipynb
```

(needs `jupyter`, `nbconvert[webpdf]` + `playwright install chromium`, `pandas` installed
into the repo's `.venv` -- these are analysis-only dependencies, not part of the main
`metawork`/`metatrain` environment's requirements.)
