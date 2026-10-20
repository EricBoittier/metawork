"""Float64 metatrain energies of chosen structures (ground truth for ggml CPU vs CUDA).

    python truth_energy.py CKPT SPLIT_DIR INDEX...
"""
import sys

import numpy as np
import torch

import metatomic.torch  # noqa: F401
from metatomic.torch import ModelOutput, System
from metatrain.pet.model import PET
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

model = PET.load_checkpoint(torch.load(sys.argv[1], map_location="cpu", weights_only=False), "finetune")
model = model.to(torch.float64).eval()
nl = model.requested_neighbor_lists()
d = sys.argv[2]
na = np.load(f"{d}/na.npy")
x = np.fromfile(f"{d}/x.bin", np.float32).reshape(-1, 3)
a = np.fromfile(f"{d}/a.bin", np.int32)
c = np.fromfile(f"{d}/c.bin", np.float32).reshape(-1, 3, 3)
for s in map(int, sys.argv[3:]):
    lo, hi = na[s], na[s + 1]
    cell = torch.tensor(c[s], dtype=torch.float64)
    sy = System(types=torch.tensor(a[lo:hi]), positions=torch.tensor(x[lo:hi], dtype=torch.float64), cell=cell,
                pbc=(cell != 0).any(dim=1))
    E = float(model([get_system_with_neighbor_lists(sy, nl)], {"energy": ModelOutput()})["energy"].block().values.sum())
    print(f"truth {s}: E {E:.6f}")
