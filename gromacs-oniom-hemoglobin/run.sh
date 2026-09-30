#!/usr/bin/env bash
# Deoxy human hemoglobin (PDB 1A3N) in TIP3P with ONIOM on four sites: each heme plus the
# side chain of its proximal histidine (His87 in alpha, His92 in beta), cut at CA-CB with
# link atoms. MM minimisation, heating and NpT, then ONIOM NVT (2 ps) and NVE (5 ps) twice:
# PET-MAD xs with the sites as one ML system (onvt, onve), and PET-OMol s with each site its
# own system, charge -2 and a quintet (omvt, omve; needs metatomic-site-groups, from the
# oniom-stress-fixes branch). Stages are skipped once their output exists.
# models/pet-omol-s-v1.0.0.pt: python -c "import upet; upet.save_upet(model='pet-omol',
#   size='s', version='1.0.0', output='models/pet-omol-s-v1.0.0.pt')" (in /.venv-upet)
set -euo pipefail
cd "$(dirname "$0")"
G=${GMX:-../gromacs-plainpairlist/build/bin/gmx}   # oniom-stress-fixes build
NT=${NT:-16}
gmx() { "$G" "$@"; }
have() { [[ -f $1 ]]; }

# charmm27.ff/aminoacids.hdb here adds hydrogen rules for HEME, which the shipped
# force field lacks; specbond.dat here links His NE2-Fe at 0.225 nm (deoxy Fe-N is
# 0.22-0.24 nm, beyond the shipped 0.2 nm +- 10%)
if ! have ions.gro; then
    grep -E '^(ATOM|HETATM)' 1A3N.pdb | grep -v HOH \
        | awk 'substr($0,17,1)==" " || substr($0,17,1)=="A"' | sed 's/^\(.\{16\}\)A/\1 /' > hb_clean.pdb
    echo END >> hb_clean.pdb
    gmx pdb2gmx -f hb_clean.pdb -ff charmm27 -water tip3p -ignh -merge all -o hb.gro -p topol.top -i posre.itp
    gmx editconf -f hb.gro -o box.gro -d 1.0 -c -bt dodecahedron
    gmx solvate -cp box.gro -cs spc216.gro -p topol.top -o solv.gro
    gmx grompp -f mdp/em.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 1   # net charge before genion
    echo SOL | gmx genion -s ions.tpr -o ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15
fi

step() {  # step NAME MDP [grompp args]: run one stage unless it is done
    have $1.gro && return
    gmx grompp -f mdp/$2.mdp -p topol.top -n index.ndx -o $1.tpr "${@:3}"
    gmx mdrun -deffnm $1 -nt "$NT"
}

if ! have index.ndx; then
    printf "q\n" | gmx make_ndx -f ions.gro -o index.ndx
fi
step em  em  -c ions.gro
step nvt nvt -c em.gro  -r em.gro
step npt npt -c nvt.gro -r em.gro -t nvt.cpt

if ! grep -q '\[ ML \]' index.ndx; then
    sel='"ML" resname HEM or (resname HIS and (resid 87 or resid 92) and not name N HN CA HA C O)'
    gmx select -s npt.tpr -f npt.gro -select "$sel" -on ml.ndx
    sed -i "1s/.*/[ ML ]/" ml.ndx
    cat ml.ndx >> index.ndx
fi
step onvt onvt -c npt.gro -t npt.cpt
step onve onve -c onvt.gro -t onvt.cpt

# one index group per site, and the whole solute
if ! grep -q '\[ SITE1 \]' index.ndx; then
    printf "0\n" | gmx trjconv -s npt.tpr -f npt.gro -o npt_whole.gro -pbc mol
    python3 make_sites.py
fi
step omvt onvt_omol -c npt.gro -t npt.cpt
step omve onve_omol -c omvt.gro -t omvt.cpt
