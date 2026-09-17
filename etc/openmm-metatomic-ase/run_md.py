"""Confirm the calculator survives repeated calls in an actual MD loop, not
just a single-point getState(). Runs NVE dynamics on the diamond toy model
and checks total energy (kinetic + potential) stays bounded across steps --
the real thing openmm-ml + MetatomicCalculator is meant for.

    .venv/bin/python run_md.py cpu
    .venv/bin/python export_toy_cuda_model.py exported-model-cuda.pt
    .venv/bin/python run_md.py cuda exported-model-cuda.pt
"""

import sys

import ase.build
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit

from metatomic_ase import MetatomicCalculator
from openmmml import MLPotential

DEFAULT_MODEL = "../../metatomic/examples/ase/exported-model.pt"


def main(device: str = "cpu", model_path: str = DEFAULT_MODEL, n_steps: int = 200) -> None:
    n_steps = int(n_steps)
    primitive = ase.build.bulk(name="C", crystalstructure="diamond", a=3.567)
    atoms = ase.build.make_supercell(primitive, 3 * np.eye(3))

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("XXX", chain)
    for _ in atoms:
        topology.addAtom("C", app.element.carbon, residue)

    calculator = MetatomicCalculator(model_path, device=device, do_gradients_with_energy=True)
    potential = MLPotential("ase")
    system = potential.createSystem(topology, calculator=calculator)
    for i in range(system.getNumParticles()):
        system.setParticleMass(i, 12.0)

    integrator = openmm.VerletIntegrator(0.5 * unit.femtoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    positions = atoms.positions + np.random.default_rng(0).normal(scale=0.02, size=atoms.positions.shape)
    context.setPositions(unit.Quantity(positions, unit.angstrom))
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
