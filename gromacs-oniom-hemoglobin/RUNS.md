# RUNS — hemoglobin ONIOM

Earlier runs (2026-09-30, kuma h100, libtorch build `oniom-stress-fixes` @ `6c40cb8162`): `tasks.txt`
(s1–s8, arrays 4482216) and `allml-tasks.txt` (4482256, 4482262, 4482378, 4482379, 4482580); logs in
`slurm-*.out` and `*.txt.d/*/log`. All tasks done; the first allml array failed and was resubmitted.

C-API rerun: GROMACS fork `capi-binding` @ `2fe524e227`, metatomic fork `torch-plugin` @ `81f97ad7`
(upstream `metatomic-core` @ `005d5950` + the torch plugin), metatensor-core 0.2.5, torch 2.13.0+cu126.
Scripts: `~/metawork/builds/build-capi.sbatch`, `~/metawork/builds/hb-capi.sbatch` (s1's MM stages
copied, the four ONIOM stages rerun in `runs/s1-capi`).

| date | job | name | cluster partition / QOS | wall | CHF bound | SHA | status | outcome | actual CHF |
|---|---|---|---|---|---|---|---|---|---|
| 09-30 | 4482547 | gmx-eccgpu-build (`builds/build-eccgpu.sbatch`) | kuma h100 / normal | 1 h | 0.52 | eccgpu build (uncommitted then, now db9b047660) | FAILED 1:23 | compile errors (vec.h include path, double precision in the kernel); fixed | ≤0.02 |
| 09-30 | 4482548 | eccgpu-test | kuma h100 / normal | 1 h | 0.52 | eccgpu build (uncommitted then, now db9b047660) | CANCELLED (dependency failed) | - | 0 |
| 09-30 | 4482551 | gmx-eccgpu-build | kuma h100 / normal | 1 h | 0.52 | eccgpu build (uncommitted then, now db9b047660) | COMPLETED 0:57 | `builds/gromacs-oniom-kuma-x86_64-eccgpu`, mixed precision, CUDA | ≤0.01 |
| 09-30 | 4482552 | eccgpu-test (`builds/eccgpu-test.sbatch`, GMX_METATOMIC_ECC_CHECK=1, 20 steps GPU and CPU) | kuma h100 / normal | 1 h | 0.52 | eccgpu build (uncommitted then, now db9b047660) | COMPLETED 1:08 | GPU vs CPU first step: allml xs (triclinic) 37,909,898 pairs both, E -7621.606 vs -7621.611 kJ/mol, max dF 0.51 of 249 kJ/mol/nm, max dvirial 0.25; sites OMol 661 pairs, 165.80256 vs 165.80257, dF 9e-6; shell NpT 52,429 pairs, 1313.542 vs 1313.543, dF 2e-4 | ≤0.01 |
| 09-30 | 4482576 | eccgpu-test2 (`builds/eccgpu-test2.sbatch`: 400-step timing GPU/CPU + 2 ps NVE on GPU, allml xs) | kuma h100 / normal | 2 h | 1.04 | eccgpu build (uncommitted then, now db9b047660) | COMPLETED 8:42 | correction 6.7 vs 262.3 ms, 122.9 vs 369.5 ms/step (3.0x); NVE 4000 steps conserved-energy drift 5.45e-4 kJ/mol/ps per atom. Outputs `builds/eccgpu-test/` | ≤0.08 |
| 10-12 | 4552855 | gmx-capi-build | kuma h100 / build (CPU) | 3 h | 0 | 20456d3eb3 / cae044fd | FAILED 1 min 21 s | torch plugin did not compile against the 8 Oct core (execute_inner takes optional<Labels>, atomic_types vector<int>); fixed in 81f97ad7 | 0 |
| 10-12 | 4552856 | hb-capi | kuma h100 / normal, 1 GPU | 1 h | 0.52 | 20456d3eb3 / cae044fd | CANCELLED (dependency never satisfied) | | 0 |
| 10-12 | 4552869 | gmx-capi-build | kuma h100 / build (CPU) | 3 h | 0 | 2fe524e227 / 81f97ad7 | COMPLETED 11 min | gmx 2fe524e227 CUDA, links libmetatomic (no libtorch); plugin built | 0 |
| 10-12 | 4552870 | hb-capi | kuma h100 / normal, 1 GPU | 1 h | 0.52 | 2fe524e227 / 81f97ad7 | COMPLETED 8.5 min (kh004) | step-0 ML energy = libtorch s1 within 0.016 kJ/mol (xs and OMol-s); NVE drift xs 0.01, OMol-s 1.41 kJ/mol/ps (libtorch s1-s8: 1.0 ± 0.3, 0.4 ± 0.6); ms/step xs 14.3/12.8 vs 16.2/15.3, OMol-s 21.5/21.1 vs 20.5/20.2 (s1 ran on kh046) | ≤0.08 |

## Whole solute on the card, and long NVE (submitted 2026-10-12 ~11:35)

Memory test (`builds/mem-test.sbatch`): allml.sh, every mdrun capped at 30 steps, six models x both
builds (libtorch `gromacs-oniom-kuma-x86_64` @ 6c40cb8162; C API `gromacs-capi-kuma-x86_64` @
2fe524e227 + plugin 81f97ad7), peak card memory from nvidia-smi every 250 ms; `runs/memtest/<partition>.tsv`.
Long NVE (`builds/hb-nve-long.sbatch BUILD MODEL 2`): 2 ns NVE (dt 0.5 fs, 4M steps, xtc every
2.5 ps) continuing each build's 5 ps NVE of s1, in 12 h segments chained afterok; `runs/s1-long/`.

| date | job | name | cluster partition / QOS | wall | CHF bound | SHA | status | outcome | actual CHF |
|---|---|---|---|---|---|---|---|---|---|
| 10-12 | 4553125 | hb-memtest | kuma h100 / debug | 1 h | 0 | both builds | COMPLETED 12.5 min (kh072; status updated 10-21) | C API: xs 10.4, s 31.1, OMol-s 29.9, OMol-m 73.7, m 95.0 (card full) GB; OMol-l OOM. libtorch: xs 9.6, s 29.2, OMol-s 28.0, OMol-m 69.7, m 92.3 GB; OMol-l OOM. ms/step (30 steps) C API 568 / 758 / 686 / 1394 / 1600, libtorch 445 / 592 / 561 / 1271 / 1440. libtorch xs marked FAILED only because allml.sh line 49 hit `ndx: command not found` after both mdruns finished. runs/memtest/h100.tsv | 0 |
| 10-12 | 4553127 | hb-memtest | kuma l40s / debug | 1 h | 0 | both builds | COMPLETED | L40S 48 GB, whole solute: xs 10.2/9.3, s 30.8/28.9, OMol-s 29.9/27.7 GB (C API/libtorch); OMol-m, m, OMol-l OOM | 0 |
| 10-12 | 4553128, 4553129 | nve-long capi xs | kuma h100 / normal | 2 x 12 h | 12.48 | 2fe524e227 / 81f97ad7 | CANCELLED after 11.5 min | cut CA-CB bonds unrestrained with link atoms (both builds); rerun on fixed builds | ≤0.10 |
| 10-12 | 4553130, 4553131 | nve-long torch xs | kuma h100 / normal | 2 x 12 h | 12.48 | 6c40cb8162 | CANCELLED after 11.5 min | cut CA-CB bonds unrestrained with link atoms (both builds); rerun on fixed builds | ≤0.10 |
| 10-12 | 4553132, 4553133, 4553135 | nve-long capi omol | kuma h100 / normal | 3 x 12 h | 18.72 | 2fe524e227 / 81f97ad7 | CANCELLED after 11.5 min | cut CA-CB bonds unrestrained with link atoms (both builds); rerun on fixed builds | ≤0.10 |
| 10-12 | 4553136, 4553137, 4553140 | nve-long torch omol | kuma h100 / normal | 3 x 12 h | 18.72 | 6c40cb8162 | CANCELLED after 11.5 min | cut CA-CB bonds unrestrained with link atoms (both builds); rerun on fixed builds | ≤0.10 |
| 10-12 | 4553261 | hb-memtest System (51,402 atoms ML, PET-MAD xs/s, CAP 10) | kuma h100 / debug | 1 h | 0 | both builds | submitted | | |

## Cut-bond fix and reruns (submitted 2026-10-12 ~11:50)

Link atoms removed the MM bond at each cut (the boundary MM atom joined the embedded set before the
topology surgery): the four cut CA-CB bonds averaged 0.20-0.23 nm (max 0.30) in s1's 5 ps NVE,
against 0.155 nm for the other histidines. Fixed on both lines: GROMACS fork `capi-binding` @
`65998f3cae` (C API) and new branch `oniom-torch-linkbond` @ `4ba569e46a` (libtorch; PR #18's head
untouched). Builds install to `builds/gromacs-{capi,torch}-linkbond-kuma-x86_64`; s1's ONIOM stages
rerun in `runs/s1-fix-{torch,capi}`; 2 ns NVE from those in `runs/s1-long/{torchfix,capifix}-{xs,omol}`.

| date | job | name | cluster partition / QOS | wall | CHF bound | SHA | status | outcome | actual CHF |
|---|---|---|---|---|---|---|---|---|---|
| 10-12 | 4553395 | gmx-linkbond-build (+ unit tests) | kuma h100 / build (CPU) | 4 h | 0 | fefbcb3ffc / 95e5489630 | FAILED 6 min | new unit test listed too few of the strict logger's messages (its bond checks passed); test amended; dependents (4553396-4553407) cancelled, never ran | 0 |
| 10-12 | 4553462 | gmx-linkbond-build (+ unit tests) | kuma h100 / build (CPU) | 4 h | 0 | 65998f3cae / 4ba569e46a | COMPLETED 10.5 min | both builds installed; metatomic and topology unit tests pass on both | 0 |
| 10-12 | 4553463 | hb-fix torch | kuma h100 / normal | 1 h | 0.52 | 4ba569e46a | COMPLETED 8.7 min | cut CA-CB 0.155 nm (0.145-0.162) | ≤0.08 |
| 10-12 | 4553464 | hb-fix capi | kuma h100 / normal | 1 h | 0.52 | 65998f3cae / 81f97ad7 | COMPLETED 8.3 min | cut CA-CB 0.156 nm (0.148-0.163) | ≤0.08 |
| 10-12 | 4553465, 4553466 | nve-long capifix xs | kuma h100 / normal | 2 x 12 h | 12.48 | 65998f3cae / 81f97ad7 | COMPLETED 11:41:11 + 1:44:24 (to 10-13 01:41) | 2 ns done (4M steps). Total energy drift (gmx energy Tot-Drift, 0-2000 ps) -4005 kJ/mol = **-2.00 kJ/mol/ps**; T 301.5 K, Tot-Drift -2.5 K; last segment (mdrun log) -3.75e-05 kJ/mol/ps per atom, 3.61 ns/day | ~7.0 |
| 10-12 | 4553467, 4553468 | nve-long torchfix xs | kuma h100 / normal | 2 x 12 h | 12.48 | 4ba569e46a | COMPLETED 11:41:04 + 3:47:49 (to 10-13 03:45) | 2 ns done. Total energy drift -3904 kJ/mol = **-1.95 kJ/mol/ps**; T 302.4 K, -2.6 K; last segment -3.85e-05 kJ/mol/ps per atom, 3.15 ns/day | ~8.0 |
| 10-12 | 4553469, 4553470, 4553471 | nve-long capifix omol | kuma h100 / normal | 3 x 12 h | 18.72 | 65998f3cae / 81f97ad7 | COMPLETED 11:41:11 + 11:41:17 + 0:00:43 (to 10-13 11:41) | 2 ns done (segment 3 ran the last 881 steps, so the log's 3.87e-04 per atom covers only those). Total energy drift -4097 kJ/mol = **-2.05 kJ/mol/ps**; T 300.4 K, -2.9 K; ~2.04 ns/day (segment 1: 1.99M steps = 0.99 ns in 11.7 h) | ~12.2 |
| 10-12 | 4553472, 4553473, 4553474 | nve-long torchfix omol | kuma h100 / normal | 3 x 12 h | 18.72 | 4ba569e46a | COMPLETED 11:41:04 + 11:26:53 + 0:00:02 (to 10-13 11:24; segment 3 had nothing left) | 2 ns done. Total energy drift -4140 kJ/mol = **-2.07 kJ/mol/ps**; T 301.6 K, -2.8 K; last segment -4.07e-05 kJ/mol/ps per atom, 1.95 ns/day | ~12.0 |
| 10-12 | 4553261 | hb-memtest System (all 51,402 atoms ML in GROMACS) | kuma h100 / debug | 1 h | 0 | 2fe524e227 | CANCELLED | mdrun segfaults at startup with every atom ML under ONIOM (both models, C API); not a memory limit | 0 |
| 10-12 | 4553460 | hb-fit-all (model only, whole system, fit_all.py) | kuma h100 / debug | 1 h | 0 | torch-cu126 venv | COMPLETED 1.5 min | all 51,402 atoms (periodic): PET-MAD xs 35.2 GiB, 0.43 s/call; PET-MAD s fits 30k (66.9 GiB), OOM at 40k; PET-OMol s fits 40k (85.7 GiB), OOM at 51k; PET-OMol m and PET-MAD m OOM at 15k. Times at small sizes inflated by JIT re-profiling | 0 |
| 10-12 | 4553536 | halo-test (build capi-halo-split d253abb1fa; whole solute on 1 vs 2 GPUs, with/without halo split; PET-MAD xs, m) | kuma h100 / debug, 2 GPUs | 1 h | 0 | d253abb1fa | FAILED (test mdp duplicated nstenergy/nstcalcenergy; build ok) | | 0 |
| 10-12 | 4553681 | halo-test (rerun, mdp fixed) | kuma h100 / debug, 2 GPUs | 1 h | 0 | d253abb1fa | COMPLETED 14 min | whole solute, one frame: halo split = 2-rank run (dE 0, xs; -0.5 of -5.65e6, m; max dF 0.40 either way). Peak per GPU, rerun: xs 9.3 -> 7.5 GB (4641 home + 2856 halo = 83%), m 84.3 -> 82.5/83.1 GB (98%, 2.55 nm range, 3 hops). 40-step MD: xs 0.080 (1 GPU) vs 0.063 ns/day (2 GPUs, either), m 0.026 vs 0.024/0.023; in MD rank 1 showed no memory saving (open) | 0 |
| 10-12 | 4554518 | hb-profile (both fixed builds side by side, PET-MAD xs and PET-OMol s, plain + GMX_METATOMIC_TIMER/METATOMIC_TORCH_TIMER, 4000 steps) | kuma h100 / debug, 2 GPUs | 1 h | 0 | 4ba569e46a / 65998f3cae / 81f97ad7 | COMPLETED 9 min (kh003) | same node: xs torch 3.07 vs capi 3.11 ns/day; OMol-s 2.12 vs 1.99. Forward identical; capi OMol-s slower in plugin input conversion (1.84 vs 0.45 ms for xs) and per-system backward/outputs | 0 |

2 ns NVE summary (10-21, from `gmx energy` on runs/s1-long/{torchfix,capifix}-{xs,omol}/nve.edr, Total Energy Tot-Drift over 0-2000 ps): xs torch -1.95, C API -2.00; omol torch -2.07, C API -2.05 kJ/mol/ps. Same slow downward drift (~-4,000 kJ/mol over 2 ns, 0.6 % of the ~-7e5 kJ/mol total; T stays at 300-302 K, -2.5 to -2.9 K) for both builds and both models, so it is not a C API vs libtorch difference. The 5 ps numbers above (xs 0.01, OMol-s 1.41 kJ/mol/ps) were too short to show it.


22 Oct (Claude): the GPU embedded Coulomb correction (tested above, 4482552 / 4482576) was only an uncommitted change in `gromacs-oniom-src`; committed as `db9b047660` on the fork's new branch `oniom-ecc-gpu` (on `oniom-stress-fixes` @ `6c40cb8162`) and pushed. Its build's version string `6c40cb8162-dirty` means this commit. Not run since: ctest, and the C API path (`capi-binding`). Also pushed: `capi-halo-split` @ `4dffd59a40` (threaded pair packing on the regular path, used by mlwater-nl single 4558456).
