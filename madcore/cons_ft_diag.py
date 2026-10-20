"""Where does the converted PET stop matching its GGUF? (conservative fine-tune gate)

    python cons_ft_diag.py GGUF START.ckpt GATE.ckpt VAL_DIR [CSV]

1. weights of START.ckpt and GATE.ckpt (after an lr 1e-12 epoch) vs the GGUF
2. composition / scales of GATE.ckpt vs the GGUF's
3. GATE's energies (eval mode, float64) on validation structures vs labels and
   vs the GGUF through mlipcpp-md (CSV from `mlipcpp-md check --csv`)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/boittier/metawork/mlip.cpp/scripts")
sys.path.insert(0, "/home/boittier/metawork/mlip.cpp/scripts/train")
from export_pytorch.export_pet_full import PETFullModel, _unwrap_to_pet, get_model_params  # noqa: E402
from export_pytorch.export_pet_gguf import metatrain_baselines  # noqa: E402
from gguf_to_checkpoint import read_gguf  # noqa: E402

import metatomic.torch  # noqa: E402,F401
from metatrain.pet.model import PET  # noqa: E402


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return PET.load_checkpoint(ckpt, "finetune")


def wrapper_params(model):
    pet = _unwrap_to_pet(model)
    p = get_model_params(pet)
    w = PETFullModel(pet, n_atoms=7, max_neighbors=11, d_pet=p["d_pet"], forces=True, cutoff=p["cutoff"],
                     cutoff_width=p["cutoff_width"], cutoff_function=p["cutoff_function"], nc_forces=True)
    return {n.replace(".", "_"): t.detach().numpy() for n, t in w.named_parameters()}


meta, tensors = read_gguf(sys.argv[1])
names = json.loads(meta["train.param_names"])
for tag, path in [("start", sys.argv[2]), ("gate", sys.argv[3])]:
    m = load(path)
    w = wrapper_params(m)
    diffs = {n: float(np.abs(w[n].reshape(-1) - tensors[n]).max()) for n in names if n in w}
    worst = sorted(diffs.items(), key=lambda kv: -kv[1])[:5]
    print(f"{tag}: {len(diffs)}/{len(names)} tensors matched by name; worst |diff| {worst}")
    print(f"  atomic types: {len(m.atomic_types)}  species_to_species_index[1,6,8]: "
          f"{m.backend.species_to_species_index[[1, 6, 8]].tolist() if hasattr(m, 'backend') else '?'}")

gate = load(sys.argv[3])
comp, e_scale, f_scales = metatrain_baselines(gate)
gcomp = dict(zip(meta["pet.composition_keys"], meta["pet.composition_values"]))
gf = dict(zip(meta.get("pet.nc_force_scale_keys", []), meta.get("pet.nc_force_scale_values", [])))
dc = max(abs(comp.get(z, np.nan) - v) for z, v in gcomp.items())
df = max(abs(f_scales.get(z, np.nan) - v) for z, v in gf.items()) if gf else 0
print(f"gate baselines: energy scale {e_scale} (GGUF {meta['pet.energy_scale']}); "
      f"max |composition diff| {dc:.3g} over {len(gcomp)} types; max |force scale diff| {df:.3g}")

# forward on validation structures
from metatomic.torch import ModelOutput, System  # noqa: E402
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists  # noqa: E402

d = Path(sys.argv[4])
na = np.load(d / "na.npy")
x = np.fromfile(d / "x.bin", np.float32).reshape(-1, 3)
a = np.fromfile(d / "a.bin", np.int32)
c = np.fromfile(d / "c.bin", np.float32).reshape(-1, 3, 3)
e = np.fromfile(d / "energy.bin", np.float32)
f = np.fromfile(d / "non_conservative_force.bin", np.float32).reshape(-1, 3)
ref = {}
if len(sys.argv) > 5:
    for row in np.genfromtxt(sys.argv[5], delimiter=",", names=True):
        ref[int(row["index"])] = row["E_engine"]
gate = gate.to(torch.float64).eval()
nl = gate.requested_neighbor_lists()
for s in range(0, min(len(na) - 1, 300), 30):
    lo, hi = na[s], na[s + 1]
    cell = torch.tensor(c[s], dtype=torch.float64)
    sy = System(types=torch.tensor(a[lo:hi]), positions=torch.tensor(x[lo:hi], dtype=torch.float64), cell=cell,
                pbc=(cell != 0).any(dim=1))
    out = gate([get_system_with_neighbor_lists(sy, nl)],
               {"energy": ModelOutput(), "non_conservative_force": ModelOutput(sample_kind="atom")})
    E = float(out["energy"].block().values.sum())
    F = out["non_conservative_force"].block().values.reshape(-1, 3).detach().numpy()
    n = hi - lo
    print(f"  val {s}: n {n}  E/atom torch {E / n:.4f}  label {e[s] / n:.4f}  mlipcpp {ref.get(s, np.nan) / n:.4f}  "
          f"F rms err torch {np.sqrt(((F - f[lo:hi]) ** 2).mean()):.3f} eV/A")
