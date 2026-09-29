# ONIOM test: cyclic peptide in a 5 nm water box

Segetalin A, cyclo(Gly-Val-Pro-Val-Trp-Ala) (87 atoms, neutral), in 4025
TIP3P waters (12162 atoms, 5 nm cubic box). The peptide is described by
PET-MAD xs v1.5.0, the water by amber99sb-ildn/TIP3P, with the subtractive
scheme `metatomic-oniom = yes`:

    E = E_MM(full) + E_PET-MAD(peptide) - E_MM(peptide)

The whole peptide is the ML region, so there are no link atoms; the
peptide-water coupling is mechanical embedding (MM LJ + MM charges).

## Binary

`../gromacs-oniom-torch` is a local worktree (branch `oniom-torch-test`,
never pushed) of `fix/oniom-subtraction` (metatensor/gromacs#11) merged with
`metatomic` (for the virial and triclinic fixes), built with Torch. The
`oniom-capi` build in `../gromacs-oniom-capi` only loads C-API models, and
there is no C-API plugin for TorchScript models such as PET-MAD yet.

## Protocol (`run.sh`)

| step | engine | length | notes |
| --- | --- | --- | --- |
| `em`  | MM | 515 steps | steepest descent |
| `nvt` | MM | 50 ps, 2 fs | V-rescale 300 K, peptide posres, h-bonds |
| `npt` | ONIOM | 20 ps, 0.5 fs | V-rescale 300 K, C-rescale 1 bar, no peptide constraints |
| `nve` | ONIOM | 20 ps, 0.5 fs | from `npt.cpt` |

`build_peptide.py` (RDKit) makes the heavy-atom structure with all-trans
peptide bonds; `specbond.dat` lets `pdb2gmx -ter` (termini: None) close the
ALA6 C - GLY1 N bond.

## Cut-offs

* PET-MAD reports an interaction range of 1.5 nm, and mdrun refuses a plain
  pairlist longer than the normal one, so `rlist` is fixed
  (`verlet-buffer-tolerance = -1`) past it.
* `rcoulomb` must exceed the largest intra-peptide distance (1.63 nm). With
  PME, the ML-ML pairs are excluded, and the exclusion correction
  `-q_i q_j erf(beta r)/r` that removes their reciprocal-space interaction is
  only applied within `rcoulomb`. With `rcoulomb = 1.0`, 469 peptide pairs lie
  beyond it: their MM Coulomb interaction stays in the reciprocal sum (double
  counted with PET-MAD), and the energy jumps each time a pair crosses the
  cut-off. `rc1.0/` keeps those runs: the NVE total energy fluctuates by
  34 kJ/mol (0.5 kJ/mol for the MM-only control, `rc1.0/nve_mm*`), the
  fluctuation does not shrink with the timestep, and finite differences of
  `Coulomb (SR)` differ from the forces by up to 800 kJ/mol/nm on single atoms
  (`fdcheck/`). With `rcoulomb = 1.9` the forces match finite differences
  within float32 noise (`fdcheck19/`).

`dtcheck/compare_ml.py` checks the reported `Metatomic Potential` against a
Python PET-MAD evaluation of the whole peptide on NVE frames (agreement
within the xtc rounding, ~4 kJ/mol).

## Results (rcoulomb = 1.9 nm, 16 OpenMP threads + RTX 4070 Ti SUPER, ~20 ms/step)

* NpT (20 ps): T = 300.4 +- 2.8 K, density 985 kg/m^3, box 4.97 nm.
* NVE (20 ps): total-energy drift +0.18 kJ/mol/ps (1.5e-5 kJ/mol/ps/atom),
  residual std 0.57 kJ/mol; T = 296.8 K. With rcoulomb = 1.0 the drift was
  -1.3 kJ/mol/ps and the fluctuation 34 kJ/mol.
* The ring stays closed (C6-N1 0.13 nm) with all-trans peptide bonds.
  PET-MAD flags one atom above the energy-uncertainty threshold on ~1% of
  the logged steps.

## With the embedded Coulomb correction (`fixed/`)

The fix (uncommitted, in `../gromacs-oniom-capi` and ported to
`../gromacs-oniom-torch`) removes the MM Coulomb interaction of every ML-ML
pair, including those beyond rcoulomb, so rcoulomb = 1.0 nm works again:

| NVE, 20 ps | drift (kJ/mol/ps) | residual std (kJ/mol) |
| --- | --- | --- |
| rcoulomb 1.0, before the fix | -1.34 | 33.9 |
| rcoulomb 1.9, before the fix | +0.18 | 0.57 |
| rcoulomb 1.0, with the fix | -0.90 | 0.54 |
| MM only, rcoulomb 1.0 | -0.96 | 0.5 |

The total Coulomb energy of one frame changes by 3.97 kJ/mol between
rcoulomb 1.0 and 1.9 with ONIOM, and by 3.99 kJ/mol for the MM-only system,
so the ML region no longer depends on the cut-off.
