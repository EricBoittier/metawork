# Hemoglobin with four ML heme sites

A stress test of the GROMACS metatomic ONIOM interface: deoxy human hemoglobin (PDB 1A3N),
CHARMM27 with TIP3P and 0.15 M NaCl (51,402 atoms), with each heme and the side chain of its
proximal histidine as a machine-learned site, cut at CA-CB with link atoms. The code fixes it
needed are on the `oniom-stress-fixes` branch of the GROMACS fork; the report is the
"Hemoglobin ONIOM Stress Test" artifact.

* `run.sh [RUN_DIR [SEED]]`: build, MM equilibration, then ONIOM NVT + NVE with PET-MAD xs (the
  sites as one system) and with PET-OMol s (one system per site, charge -2, quintet).
  `tasks.txt` runs eight seeds; submit it with `etc/oniom/submit.sh tasks.txt` on any of the
  clusters (see `etc/oniom/README.md`). Paths come from `etc/oniom/env.sh`.
* `allml.sh RUN_DIR MODEL`: the whole solute (9,026 atoms) as ML from a replica's NpT, the same
  NVT + NVE protocol; PET-OMol gets it as one system, charge -6, multiplicity 1 (the four high-spin
  Fe(II) paired antiparallel; 17, all aligned, is outside PET-OMol's 1-11). `allml-tasks.txt` runs PET-MAD xs, s, m and PET-OMol s, m on replica s1.
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

## GPU memory with the whole solute as ML (H100, 94 GB)

`models/fit_test.py` on growing parts of the solute: energy + forces, time of the second call.
Memory grows linearly with the ML atoms; the last column extrapolates to about 90 GB.

| Model | 9,026 atoms | peak | per atom | largest ML region |
|---|---|---|---|---|
| PET-MAD xs v1.5.0 | 0.073 s | 5.5 GiB | 0.6 MiB | ~150k atoms |
| PET-MAD s v1.5.0 | 0.23 s | 18.8 GiB | 2.1 MiB | ~43k |
| PET-MAD m v1.6.0 | 1.03 s | 75.5 GiB | 8.4 MiB | ~11k |
| PET-OMol s v1.0.0 | 0.23 s | 18.1 GiB | 2.0 MiB | ~45k |
| PET-OMol m v1.0.0 | 0.87 s | 61.8 GiB | 6.9 MiB | ~13k |
| PET-OMol l v1.0.0 | out of memory (1.08 s, 64.9 GiB at 6,000) | | 10.8 MiB | ~8k |
