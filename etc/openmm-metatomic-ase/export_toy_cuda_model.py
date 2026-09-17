"""Export a tiny CUDA-capable metatomic model for smoke-testing device="cuda".

`metatomic/examples/ase/exported-model.pt` is CPU-only (its
ModelCapabilities.supported_devices == ["cpu"]), so it can't be used to check
the device="cuda" path end to end. This is the same Einstein-solid toy model
from that tutorial, just declaring cuda support and with equilibrium_positions
registered as a buffer so `.to(device)` actually moves it.
"""

from typing import Dict, List, Optional

import ase.build
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)


class HarmonicModel(torch.nn.Module):
    def __init__(self, force_constant: float, equilibrium_positions: torch.Tensor):
        super().__init__()
        assert force_constant > 0
        self.force_constant = force_constant
        self.register_buffer("equilibrium_positions", equilibrium_positions)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        device = systems[0].positions.device
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype, device=device)
        for i, system in enumerate(systems):
            assert len(system) == self.equilibrium_positions.shape[0]
            r0 = self.equilibrium_positions
            energy[i] += torch.sum(self.force_constant * (system.positions - r0) ** 2)

        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(len(systems), device=device).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]], device=device)),
        )
        return {"energy": TensorMap(Labels("_", torch.tensor([[0]], device=device)), [block])}


def export(output: str = "exported-model-cuda.pt") -> None:
    primitive = ase.build.bulk(name="C", crystalstructure="diamond", a=3.567)
    atoms = ase.build.make_supercell(primitive, 3 * np.eye(3))

    model = HarmonicModel(
        force_constant=3.14159265358979323846,
        equilibrium_positions=torch.tensor(atoms.positions, dtype=torch.float32),
    )
    capabilities = ModelCapabilities(
        outputs={"energy": ModelOutput(unit="eV", sample_kind="system")},
        atomic_types=[6],
        interaction_range=0.0,
        length_unit="Angstrom",
        supported_devices=["cuda", "cpu"],
        dtype="float32",
    )
    wrapper = AtomisticModel(model.eval(), ModelMetadata(), capabilities)
    wrapper.save(output)
    print(f"exported {output} ({len(atoms)} atoms)")


if __name__ == "__main__":
    import sys

    export(*sys.argv[1:])
