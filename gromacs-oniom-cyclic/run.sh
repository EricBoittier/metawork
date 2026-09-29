#!/usr/bin/env bash
# Cyclic peptide (segetalin A) in a 5 nm TIP3P box with ONIOM PET-MAD/amber99sb-ildn:
# MM minimisation and heating, then ONIOM NpT (20 ps) and ONIOM NVE (20 ps).
set -euo pipefail
cd "$(dirname "$0")"
G=${GMX:-../gromacs-oniom-torch/build/bin/gmx}   # Torch build of fix/oniom-subtraction + metatomic
NT=${NT:-16}

# peptide.pdb from build_peptide.py (RDKit, needs rdkit): heavy atoms, all-trans omegas
printf "0\n0\n" | $G pdb2gmx -f segetalinA_heavy.pdb -ff amber99sb-ildn -water tip3p -ignh -ter \
    -o pep.gro -p topol.top -i posre.itp        # specbond.dat closes ALA6 C - GLY1 N
$G editconf -f pep.gro -o box.gro -box 5.0 -c -bt cubic
$G solvate -cp box.gro -cs spc216.gro -p topol.top -o solv.gro
printf "q\n" | $G make_ndx -f solv.gro -o index.ndx

step() { $G grompp -f $1.mdp -p topol.top -n index.ndx -o $1.tpr "${@:2}" && $G mdrun -deffnm $1 -nt $NT; }
step em  -c solv.gro
step nvt -c em.gro  -r em.gro
# rcoulomb = rlist = 1.9 nm in npt.mdp and nve.mdp (README: Cut-offs)
step npt -c nvt.gro -t nvt.cpt
step nve -c npt.gro -t npt.cpt
