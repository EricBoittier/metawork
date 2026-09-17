"""Run the openmm-ml "ase" potential + metatomic_ase.MetatomicCalculator combo
against the Einstein-solid toy model, non-periodic 54-atom diamond supercell
(same structure `exported-model.pt` was fit to; see
metatomic/examples/ase/1-md.py). Defaults to the CPU-only tutorial checkpoint;
pass the CUDA-capable one from export_toy_cuda_model.py to check device="cuda".

    .venv/bin/python run_toy_model.py cpu
    python export_toy_cuda_model.py exported-model-cuda.pt
    .venv/bin/python run_toy_model.py cuda exported-model-cuda.pt
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


def main(device: str = "cpu", model_path: str = DEFAULT_MODEL) -> None:
    primitive = ase.build.bulk(name="C", crystalstructure="diamond", a=3.567)
    atoms = ase.build.make_supercell(primitive, 3 * np.eye(3))

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("XXX", chain)
    for _ in atoms:
        topology.addAtom("C", app.element.carbon, residue)
    # non-periodic on purpose: this model compares each atom to a fixed
    # reference position, it doesn't need PBC/neighbor lists.

    calculator = MetatomicCalculator(
        model_path,
        device=device,
        do_gradients_with_energy=True,
    )
    potential = MLPotential("ase")
    system = potential.createSystem(topology, calculator=calculator)

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(unit.Quantity(atoms.positions, unit.angstrom))
    state = context.getState(getEnergy=True, getForces=True)

    print("device:", device)
    print("energy:", state.getPotentialEnergy())
    print("max |force|:", np.abs(state.getForces(asNumpy=True)).max())


if __name__ == "__main__":
    main(*sys.argv[1:])
