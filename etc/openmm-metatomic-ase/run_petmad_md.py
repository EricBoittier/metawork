"""Real MD, not a single getState() call: NVE dynamics on a water molecule
with PET-MAD, checking total energy (kinetic + potential) stays bounded
across steps.

    .venv/bin/mtt export lab-cosmo/pet-mad models/pet-mad-dev.ckpt -o exported-petmad.pt
    .venv/bin/python run_petmad_md.py exported-petmad.pt cpu
    .venv/bin/python run_petmad_md.py exported-petmad.pt cuda
"""

import sys

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit

from metatomic_ase import MetatomicCalculator
from openmmml import MLPotential


def main(model_path: str = "exported-petmad.pt", device: str = "cpu", n_steps: int = 200) -> None:
    n_steps = int(n_steps)
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HOH", chain)
    topology.addAtom("O", app.element.oxygen, residue)
    topology.addAtom("H1", app.element.hydrogen, residue)
    topology.addAtom("H2", app.element.hydrogen, residue)

    calculator = MetatomicCalculator(model_path, device=device, do_gradients_with_energy=True)
    potential = MLPotential("ase")
    system = potential.createSystem(topology, calculator=calculator)

    integrator = openmm.VerletIntegrator(0.5 * unit.femtoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(unit.Quantity(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.096], [0.0, 0.093, -0.024]],
        unit.nanometers,
    ))
    context.setVelocitiesToTemperature(300 * unit.kelvin, 0)

    energies = []
    for step in range(n_steps):
        state = context.getState(getEnergy=True)
        total = (state.getKineticEnergy() + state.getPotentialEnergy()).value_in_unit(unit.kilojoules_per_mole)
        energies.append(total)
        integrator.step(1)

    energies = np.array(energies)
    print("device:", device)
    print("steps:", n_steps)
    print("total energy: mean", energies.mean(), "std", energies.std(), "drift", energies[-1] - energies[0])
    assert np.all(np.isfinite(energies)), "energy went non-finite -- the model or integrator diverged"


if __name__ == "__main__":
    main(*sys.argv[1:])
