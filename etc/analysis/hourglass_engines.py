"""
Per-engine NVE runners for `17_hourglass_nve_lorem_pet.ipynb`.

Adapted from the upstream `metatomic-hourglass` cookbook recipe
(`metawork/atomistic-cookbook/examples/metatomic-hourglass/metatomic-hourglass.py`),
which runs PET-MAD/MACE-OMAT/DPA3 through five engines (ASE, LAMMPS, GROMACS,
i-PI, TorchSim) and checks that NVE trajectories from a zero-velocity start
agree across engines. This module keeps the same protocol (VelocityVerlet /
`nve` motion, 0.5 fs timestep, 100 steps, zero initial velocities, same 101
snapshots from t=0 to t=50 fs) but applies it to this project's own SN2 model
family instead, with four of the five engines now usable in `metawork/.venv`
on this machine: ASE, i-PI, LAMMPS (`metawork/lammps`, branch `metatomic`,
`PKG_ML-METATOMIC`), and GROMACS (`metawork/gromacs`, branch `metatomic`,
`GMX_METATOMIC=TORCH`) -- both built specifically for this notebook. Only
TorchSim still isn't installed.

GROMACS matches ASE/i-PI/LAMMPS well for the short-range PET model (~0.02 meV)
but is impractically slow for LOREM: a single 100-step run didn't finish in
10+ minutes even at the smallest box tried (20 Å) -- worse than either of the
other two periodic engines' own box-size walls (see notebook 17, sections 1
and 3). `run_gromacs` is kept and used for PET; LOREM+GROMACS is documented
as a limitation, not attempted as part of the routine comparison.

Unlike the upstream recipe's neutral ethanol molecule, the shared reference
structure here (`data/sn2/sn2.xyz`, frame 0) is a charged (-1), non-periodic
6-atom complex (CH3F...I-) -- the same structure notebooks 07/09 use. Net
charge is not passed to the calculator explicitly; it isn't in the upstream
recipe either, and MetatomicCalculator has nothing to receive it through --
these models were trained on charged SN2 species directly, charge is not a
runtime input.
"""

from pathlib import Path
from typing import List, Literal, Tuple

DT_FS = 0.5
N_STEPS = 100
# Å, edge length of the vacuum cell i-PI needs (ASE stays non-periodic). 150 Å is the
# largest size in the box-size scan below (see notebook 17) that still runs in a
# practical time -- i-PI's Ewald k-space cost grows much faster than 20-150 Å's
# ~1/L error decay, so 300 Å already took >180s for a single 100-step run (vs. a few
# tens of seconds at 150 Å) with no realistic prospect of reaching sub-meV agreement
# this way. 150 Å is the best *practically reachable* value, not a converged one.
DEFAULT_CELL = 150.0


def run_ase(
    model_path: str, atoms, ensemble: Literal["nve", "nvt"] = "nve"
) -> Tuple[List[float], List[float]]:
    import ase.md
    import ase.units
    from ase.constraints import FixCom
    from metatomic_ase import MetatomicCalculator

    atoms = atoms.copy()
    atoms.calc = MetatomicCalculator(model_path)
    atoms.set_constraint(FixCom())

    if ensemble == "nve":
        integrator = ase.md.VelocityVerlet(atoms, timestep=DT_FS * ase.units.fs)
    else:
        integrator = ase.md.Langevin(
            atoms,
            timestep=DT_FS * ase.units.fs,
            temperature_K=300,
            friction=0.01 / ase.units.fs,
            fixcm=False,
        )

    times, energies = [0.0], [atoms.get_potential_energy()]
    for step in range(N_STEPS):
        integrator.run(1)
        times.append((step + 1) * DT_FS)
        energies.append(atoms.get_potential_energy())

    return times, energies


def run_ipi(
    model_path: str, atoms, ensemble: Literal["nve", "nvt"] = "nve", cell: float = DEFAULT_CELL,
    template_xyz: str = None,
) -> Tuple[List[float], List[float]]:
    """`cell` is the edge length (Å) of the cubic vacuum box i-PI runs in --
    i-PI always treats its cell as periodic, unlike ASE's `run_ase` above, so
    this is exposed as a parameter to study how engine agreement depends on
    box size for models with a genuine long-range term (see notebook 17).
    `template_xyz` must have the same chemical symbols/ordering as `atoms`;
    it defaults to `TEMPLATE_XYZ` (the 6-atom SN2 complex) but must be
    overridden for any other structure, such as notebook 20's 5-atom neutral
    CH3F fragment."""
    import ase.units
    from ipi.scripting import (
        InteractiveSimulation,
        forcefield_xml,
        motion_nvt_xml,
        read_output,
        simulation_xml,
    )
    from ipi.utils.softexit import softexit

    structure = atoms.copy()
    structure.cell = [cell, cell, cell]

    if ensemble == "nve":
        motion = f"""
        <motion mode='dynamics'>
            <dynamics mode='nve'>
                <timestep units='femtosecond'> {DT_FS} </timestep>
            </dynamics>
        </motion>
        """
        temperature = None
    else:
        motion = motion_nvt_xml(timestep=DT_FS * ase.units.fs)
        temperature = 300

    output = """
    <output prefix='simulation'>
        <properties stride='1' filename='out'>
            [ step, time{picosecond}, potential{electronvolt} ]
        </properties>
    </output>
    """

    # unique per (model, ensemble, box size) so repeated calls in the same
    # process/directory (e.g. the box-size scan below) don't clobber each
    # other's output files
    model_tag = Path(model_path).parent.name
    prefix = f"sn2-ipi-{model_tag}-{ensemble}-box{cell:g}"

    input_xml = simulation_xml(
        structures=structure,
        forcefield=forcefield_xml(
            name="metatomic",
            mode="direct",
            pes="metatomic",
            parameters=f"{{template:{template_xyz or TEMPLATE_XYZ},model:{model_path},device:cpu}}",
        ),
        motion=motion,
        temperature=temperature,
        output=output,
        prefix=prefix,
    )

    # softexit is global: if anything trips it, every later simulation in this
    # process quits after a few steps without saying anything
    softexit.reset()

    sim = InteractiveSimulation(input_xml)
    sim.run(N_STEPS)

    results, _info = read_output(f"{prefix}.out")
    return (
        (results["time"] * 1000).tolist(),  # ps -> fs
        results["potential"].tolist(),  # already in eV
    )


def run_lammps(
    model_path: str, atoms, ensemble: Literal["nve", "nvt"] = "nve", cell: float = DEFAULT_CELL
) -> Tuple[List[float], List[float]]:
    """Like i-PI, LAMMPS always treats its box as periodic, so `cell` has the same
    role as in `run_ipi`. Requires the `lmp` binary built with the ML-METATOMIC
    package (`metawork/lammps`, branch `metatomic`) -- see `LMP_BINARY` below."""
    import subprocess

    import ase.io
    import numpy as np
    from ase.data import atomic_numbers

    structure = atoms.copy()
    structure.cell = [cell, cell, cell]

    # species must be written in a fixed order so the `pair_coeff` line (atomic
    # numbers in that same order) maps LAMMPS atom types to the right elements
    species = sorted(set(structure.get_chemical_symbols()), key=lambda s: atomic_numbers[s])

    model_tag = Path(model_path).parent.name
    tag = f"sn2-lammps-{model_tag}-{ensemble}-box{cell:g}"
    data_file = f"{tag}.data"
    input_file = f"{tag}.in"
    output_file = f"{tag}.out"

    ase.io.write(data_file, structure, format="lammps-data", masses=True, specorder=species)
    numbers = " ".join(str(atomic_numbers[s]) for s in species)

    if ensemble == "nve":
        ensemble_setup = "velocity all zero linear\nfix 1 all nve\n"
    else:
        ensemble_setup = (
            "velocity all create 300 87287 mom yes rot yes\n"
            "fix 1 all nvt temp 300 300 0.05\n"
        )

    with open(input_file, "w") as f:
        f.write(f"""\
units metal
atom_style atomic

read_data {data_file}

pair_style metatomic {model_path} device cpu
pair_coeff * * {numbers}

neighbor 2.0 bin

timestep {DT_FS / 1000}

{ensemble_setup}
fix 2 all print 1 "$(time) $(pe)" file {output_file} screen no

run {N_STEPS}
""")

    subprocess.run(
        [LMP_BINARY, "-in", input_file, "-log", "none"],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    time_ps, pe = np.loadtxt(output_file, skiprows=1, unpack=True)
    return (time_ps * 1000).tolist(), pe.tolist()  # ps -> fs


LMP_BINARY = "/home/boittier/metawork/lammps/build/lmp"


def run_gromacs(
    model_path: str, atoms, ensemble: Literal["nve", "nvt"] = "nve", cell: float = DEFAULT_CELL
) -> Tuple[List[float], List[float]]:
    """Like i-PI and LAMMPS, GROMACS always treats its box as periodic. Requires
    the `gmx` binary built with `GMX_METATOMIC=TORCH` (`metawork/gromacs`, branch
    `metatomic`) -- see `GMX_BINARY` below. Topology is inert (masses only, no
    charges/LJ, generalizing the upstream recipe's `data/topol.top` to whatever
    species `atoms` actually has) -- all interactions come from the model."""
    import shutil
    import subprocess

    import ase.units
    from ase.data import atomic_masses, atomic_numbers

    structure = atoms.copy()
    cell_nm = cell / 10.0  # Å -> nm

    model_tag = Path(model_path).parent.name
    tag = f"sn2-gmx-{model_tag}-{ensemble}-box{cell:g}"
    group_name = "SN2"

    symbols = structure.get_chemical_symbols()
    species = sorted(set(symbols), key=lambda s: atomic_numbers[s])

    # .gro coordinate file (positions in nm)
    with open(f"{tag}.gro", "w") as f:
        f.write(f"{group_name}\n{len(structure)}\n")
        for i, (sym, pos) in enumerate(zip(symbols, structure.positions)):
            x, y, z = pos / 10.0  # Å -> nm
            f.write(f"{1:5d}{group_name:<5s}{sym:>5s}{i + 1:5d}{x:10.5f}{y:10.5f}{z:10.5f}\n")
        f.write(f"{cell_nm:10.5f}{cell_nm:10.5f}{cell_nm:10.5f}\n")

    # inert topology: masses only, no charges/LJ -- all forces from the model
    atomtypes = "\n".join(
        f"{s:6s} {atomic_numbers[s]:<6d} {atomic_masses[atomic_numbers[s]]:<8.3f} "
        f"0.000    A       0.0     0.0" for s in species
    )
    mol_atoms = "\n".join(
        f"{i + 1:4d}  {sym:<4s}  1      {group_name:<4s} {sym}{i + 1:<3d} {i + 1:4d}     0.000  "
        f"{atomic_masses[atomic_numbers[sym]]:.3f}"
        for i, sym in enumerate(symbols)
    )
    with open(f"{tag}.top", "w") as f:
        f.write(f"""\
[ defaults ]
1               2               yes             0.5     0.5

[ atomtypes ]
{atomtypes}

[ moleculetype ]
{group_name}           0

[ atoms ]
{mol_atoms}

[ system ]
{group_name} in vacuum

[ molecules ]
{group_name}           1
""")

    with open(f"{tag}.ndx", "w") as f:
        f.write(f"[ {group_name} ]\n")
        f.write(" ".join(str(i + 1) for i in range(len(structure))) + "\n")

    if ensemble == "nve":
        thermostat = "tcoupl = no"
    else:
        thermostat = f"tcoupl = v-rescale\ntc-grps = {group_name}\ntau-t = 0.1\nref-t = 300"

    # pairlist only needs to cover each model's own local/short-range featurizer
    # cutoff, not a genuinely unbounded long-range term (the model's own Ewald-type
    # evaluation runs independent of GROMACS's Verlet list) -- capped well under
    # half the box edge, as minimum-image convention requires
    rlist_nm = min(1.6, 0.45 * cell_nm)

    with open(f"{tag}.mdp", "w") as f:
        f.write(f"""\
integrator = md-vv
dt = {DT_FS / 1000}
nsteps = {N_STEPS}

cutoff-scheme = Verlet
pbc = xyz
rlist = {rlist_nm}
rcoulomb = {rlist_nm}
rvdw = {rlist_nm}

{thermostat}

metatomic-active = yes
metatomic-model = {model_path}
metatomic-input-group = {group_name}
metatomic-device = cpu

nstenergy = 1
nstlog = {N_STEPS}
""")

    gmx = shutil.which("gmx_mpi") or shutil.which("gmx") or GMX_BINARY
    for command in [
        [gmx, "grompp", "-f", f"{tag}.mdp", "-c", f"{tag}.gro", "-p", f"{tag}.top",
         "-n", f"{tag}.ndx", "-o", f"{tag}.tpr", "-maxwarn", "2"],
        [gmx, "mdrun", "-deffnm", tag],
    ]:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run(
        [gmx, "energy", "-f", f"{tag}.edr", "-o", f"{tag}_energy.xvg"],
        input="Potential\n", text=True, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    import numpy as np

    data = np.loadtxt(f"{tag}_energy.xvg", comments=["@", "#"])
    time_fs = data[:, 0] * 1000  # ps -> fs
    energy_ev = data[:, 1] * ase.units.kJ / ase.units.mol  # kJ/mol -> eV
    return time_fs.tolist(), energy_ev.tolist()


GMX_BINARY = "/home/boittier/metawork/gromacs/build/bin/gmx"

# path i-PI's direct-mode metatomic driver uses as an atom-type/species
# template; the actual geometry and cell come from `structures=structure`
# above, this only needs to match chemical symbols and ordering.
TEMPLATE_XYZ = "sn2_frame0.xyz"
