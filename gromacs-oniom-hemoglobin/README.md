# Hemoglobin with four ML heme sites

A stress test of the GROMACS metatomic ONIOM interface: deoxy human hemoglobin (PDB 1A3N),
CHARMM27 with TIP3P and 0.15 M NaCl (51,402 atoms), with each heme and the side chain of its
proximal histidine as a machine-learned site, cut at CA-CB with link atoms. The code fixes it
needed are on the `oniom-stress-fixes` branch of the GROMACS fork; the report is the
"Hemoglobin ONIOM Stress Test" artifact.

* `run.sh`: build, MM equilibration, then ONIOM NVT + NVE with PET-MAD xs (the sites as one
  system) and with PET-OMol s (one system per site, charge -2, quintet).
* `charmm27.ff/aminoacids.hdb`: adds hydrogen rules for `HEME`, which the shipped CHARMM27
  lacks; `specbond.dat` links His NE2-Fe at 0.225 nm (the deoxy distances, 0.223-0.236 nm,
  fall outside the shipped 0.2 nm +- 10%).
* `make_sites.py`: the SITE1-4 and SOLUTE index groups.
* `mdp/`: MM stages, ONIOM stages for both models, and a zero-step rerun.
* `models/`: model comparisons on isolated heme sites (`site_test.py`, `relax_sites.py`,
  relaxed geometries in `relaxed_site1/`), the per-site reference energy (`ref_sites.py`),
  one system vs four (`combined_test.py`) and GPU memory vs ML size (`fit_test.py`). The
  exported models (2 GB) are not tracked; export them with `upet.save_upet`.
* `forcecheck/`: GROMACS model forces against per-site ASE forces (A: PET-OMol on, B: off).
* `timeseries.py` -> `timeseries.json`: energies, temperature and Fe-NE2 distances.
* `render/hb_scenes.py`: POV-Ray scenes (uses `../gromacs-oniom-cyclic/render/povlib.py`).
