#!/usr/bin/env bash
# Deoxy human hemoglobin (PDB 1A3N) in TIP3P with ONIOM on four sites: each heme plus the
# side chain of its proximal histidine (His87 in alpha, His92 in beta), cut at CA-CB with
# link atoms. MM minimisation, heating and NpT, then ONIOM NVT (2 ps) and NVE (5 ps) twice:
# PET-MAD xs with the sites as one ML system (onvt, onve), and PET-OMol s with each site its
# own system, charge -2 and a quintet (omvt, omve; needs metatomic-site-groups, from the
# oniom-stress-fixes branch). Stages are skipped once their output exists, and mdrun resumes
# from its checkpoint, so the script can be restarted at any point.
#
#   ./run.sh [RUN_DIR [SEED]]   run in RUN_DIR (default: here), with SEED for the MM heating
#                               velocities (default 1); replicas can run side by side
# Paths (GMX, ONIOM_MODELS, NT) come from ../etc/oniom/env.sh; see ../etc/oniom/README.md.
# models/pet-omol-s-v1.0.0.pt: python -c "import upet; upet.save_upet(model='pet-omol',
#   size='s', version='1.0.0', output='models/pet-omol-s-v1.0.0.pt')" (in /.venv-upet)
set -euo pipefail
HB=$(cd "$(dirname "$0")" && pwd)
. "$HB/../etc/oniom/env.sh"
RUN=${1:-$HB}
SEED=${2:-1}
mkdir -p "$RUN" && cd "$RUN"
for f in 1A3N.pdb specbond.dat charmm27.ff make_sites.py; do   # shared inputs of a replica
    [[ -e $f ]] || ln -s "$HB/$f" .
done
gmx() { "$GMX" "$@"; }
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
    gmx grompp -f "$HB/mdp/em.mdp" -c solv.gro -p topol.top -o ions.tpr -maxwarn 1   # net charge before genion
    echo SOL | gmx genion -s ions.tpr -o ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15
fi

step() {  # step NAME MDP [grompp args]: run one stage unless it is done, resuming from its checkpoint
    have $1.gro && return
    oniom_mdp "$HB/mdp/$2.mdp" $1.mdp   # fills in the model path
    [[ $2 == nvt ]] && sed -i "s/^gen-seed .*/gen-seed                = $SEED/" $1.mdp
    [[ -f $1.tpr ]] || gmx grompp -f $1.mdp -p topol.top -n index.ndx -o $1.tpr -po $1.mdout.mdp "${@:3}"
    local cpi=()
    [[ -f $1.cpt ]] && cpi=(-cpi $1.cpt)
    gmx mdrun -deffnm $1 -nt "$NT" "${cpi[@]}"
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
