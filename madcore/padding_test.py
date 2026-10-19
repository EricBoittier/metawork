#!/usr/bin/env python3
"""Do padded neighbour slots explain mlip.cpp vs metatrain? (float64 metatrain)

    padding_test.py --ckpt model.ckpt --predictions predictions.npz [--n 200]

metatrain fills padded NEF slots with edge data (edge_array_to_nef without a
mask) and the attention bias is log(clamp(cutoff factor, 1e-15)) = -34.5 there,
so padded slots keep some weight. Variants, per structure (energy, direct forces):
  base   metatrain as is
  zero   padded vectors / neighbour species / neighbour index zero-filled, as mlip.cpp
  mask   attention epsilon 1e-300: padded slots excluded (bias ~ -690)
Each is compared with mlip.cpp's GraphModel CPU predictions from the npz.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/boittier/metawork/mlip.cpp/scripts")
from roundtrip_compare import memmap, torch_eval  # noqa: E402

import metatrain.pet.modules.structures as S  # noqa: E402
from metatrain.pet.modules.transformer import AttentionBlock  # noqa: E402

MODE = {"zero": False}
_get, _to_nef = S.get_nef_indices, S.edge_array_to_nef
_mask = {}


def get_nef_indices(*a, **k):
    out = _get(*a, **k)
    _mask["m"] = out[2]
    return out


def edge_array_to_nef(edge_array, nef_indices, mask=None, fill_value=0.0):
    if mask is None and MODE["zero"]:
        return _to_nef(edge_array, nef_indices, _mask["m"], 0)
    return _to_nef(edge_array, nef_indices, mask, fill_value)


S.get_nef_indices, S.edge_array_to_nef = get_nef_indices, edge_array_to_nef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split", default="/home/boittier/data/madcore/memmap-2pow24/test")
    ap.add_argument("--worst", default="162636,103022,47112,107003,2496")
    ap.add_argument("--n", type=int, default=200, help="evenly spaced structures of the npz besides --worst")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from export_pytorch.export_pet_full import load_local_model

    p = np.load(a.predictions)
    ids_all, n_all = p["ids"], p["n_atoms"]
    starts = np.r_[0, np.cumsum(n_all)[:-1]]
    worst = [int(s) for s in a.worst.split(",")]
    pick = sorted(set(np.flatnonzero(np.isin(ids_all, worst)))
                  | set(np.linspace(0, len(ids_all) - 1, a.n).round().astype(int)))
    ids, n = ids_all[pick], n_all[pick]
    atoms = np.concatenate([np.arange(starts[k], starts[k] + n_all[k]) for k in pick])
    E_c, F_c = p["E:GraphModel_CPU"][pick], p["Fn:GraphModel_CPU"][atoms]
    data = memmap(a.split)
    model = load_local_model(a.ckpt)
    attn = [m for m in model.modules() if isinstance(m, AttentionBlock)]
    print(f"{len(ids)} structures ({n.min()}-{n.max()} atoms), {len(attn)} attention blocks, epsilon {attn[0].epsilon}")

    res = {}
    for name in ["base", "zero", "mask"]:
        MODE["zero"] = name == "zero"
        for m in attn:
            m.epsilon = 1e-300 if name == "mask" else 1e-15
        E, Fn, _ = torch_eval(model, data, ids, torch.float64, a.device, conservative=False)
        res[name] = (E, Fn)
    MODE["zero"] = False

    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    print("\nrms / max differences; E meV/atom, direct F meV/A")
    for x, y in [("base", "mlip.cpp"), ("zero", "mlip.cpp"), ("mask", "mlip.cpp"), ("mask", "base")]:
        Ex, Fx = res[x]
        Ey, Fy = (E_c, F_c) if y == "mlip.cpp" else res[y]
        dE, dF = 1e3 * (Ex - Ey) / n, 1e3 * (Fx - Fy)
        print(f"  {x:5s} vs {y:9s}  E {rms(dE):9.4g} / {np.abs(dE).max():9.4g}   F {rms(dF):9.4g} / {np.abs(dF).max():9.4g}")
    print("\nworst structures, E - mlip.cpp (meV/atom): base / zero / mask")
    for s in worst:
        k = int(np.flatnonzero(ids == s)[0])
        print(f"  {s:7d} ({n[k]:3d} atoms): " + " / ".join(
            f"{1e3 * (res[v][0][k] - E_c[k]) / n[k]:+.4f}" for v in ["base", "zero", "mask"]))


if __name__ == "__main__":
    main()
