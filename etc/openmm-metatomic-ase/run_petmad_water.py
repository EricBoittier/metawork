"""The exact snippet under test, run against a real exported model: PET-MAD
(lab-cosmo/pet-mad) on a single water molecule.

Export the checkpoint first:

    .venv/bin/mtt export lab-cosmo/pet-mad models/pet-mad-dev.ckpt -o exported-petmad.pt
    .venv/bin/python run_petmad_water.py exported-petmad.pt cpu
    .venv/bin/python run_petmad_water.py exported-petmad.pt cuda
"""

import sys

import openmm
import openmm.app as app
import openmm.unit as unit

from metatomic_ase import MetatomicCalculator
from openmmml import MLPotential


def main(model_path: str = "exported-petmad.pt", device: str = "cpu") -> None:
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HOH", chain)
    topology.addAtom("O", app.element.oxygen, residue)
    topology.addAtom("H1", app.element.hydrogen, residue)
    topology.addAtom("H2", app.element.hydrogen, residue)

    calculator = MetatomicCalculator(
        model_path,
        device=device,
        do_gradients_with_energy=True,
    )

    potential = MLPotential("ase")
    system = potential.createSystem(
        topology,
        calculator=calculator,
    )

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(unit.Quantity(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.096], [0.0, 0.093, -0.024]],
        unit.nanometers,
    ))
    state = context.getState(getEnergy=True, getForces=True)

    print("device:", device)
    print("energy:", state.getPotentialEnergy())
    print("forces:", state.getForces(asNumpy=True))


if __name__ == "__main__":
    main(*sys.argv[1:])
