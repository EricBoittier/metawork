#!/usr/bin/env python3
"""Inputs or execution? metatrain's network on mlip.cpp's NEF inputs (float64).

    input_swap.py --ckpt model.ckpt --dir DIR IDX [IDX ...]

DIR/IDX.bin is `mlipcpp-md edges ... --dump-inputs` (the engine's graph inputs),
DIR/IDX.txt its printout (GraphModel CPU energy). Per structure:
  E metatrain            its own preprocessing
  E swap                 its network on mlip.cpp's inputs (cutoff factors from
                         mlip.cpp's pair cutoffs with metatrain's bump)
  E mlip.cpp             GraphModel CPU
If swap ~ mlip.cpp the inputs differ, if swap ~ metatrain ggml executes the graph
differently. Then the inputs edge by edge (matched per centre by species and
vector), and metatrain with only the cutoff factors taken from mlip.cpp.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/boittier/metawork/mlip.cpp/scripts")
from roundtrip_compare import memmap  # noqa: E402

from metatrain.pet.modules.utilities import cutoff_func_bump  # noqa: E402


def read_inputs(path):
    raw = Path(path).read_bytes()
    n, M = np.frombuffer(raw, np.int32, 2)
    off = 8
    def take(dtype, count, shape):
        nonlocal off
        a = np.frombuffer(raw, dtype, count, off).reshape(shape)
        off += a.nbytes
        return a
    out = dict(species=take(np.int32, n, (n,)), neighbor_species=take(np.int32, n * M, (n, M)),
               reverse=take(np.int32, n * M, (n, M)), edge_vectors=take(np.float32, 3 * n * M, (n, M, 3)),
               padding_mask=take(np.float32, n * M, (n, M)), cutoff_values=take(np.float32, n * M, (n, M)))
    assert off == len(raw), (off, len(raw))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--split", default="/home/boittier/data/madcore/memmap-2pow24/test")
    ap.add_argument("--eps", type=float, default=1e-15, help="attention epsilon (1e-300: padded slots excluded)")
    ap.add_argument("ids", nargs="+", type=int)
    a = ap.parse_args()
    from export_pytorch.export_pet_full import load_local_model
    from metatomic.torch import ModelOutput, System
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

    model = load_local_model(a.ckpt).to(torch.float64).eval()
    width = float(model.hypers["cutoff_width"])
    from metatrain.pet.modules.transformer import AttentionBlock
    for m in model.modules():
        if isinstance(m, AttentionBlock):
            m.epsilon = a.eps
    print(f"attention epsilon {a.eps}")
    backend = model.backend
    preprocess = backend.preprocess
    state = {}

    def hooked(*args, **kw):
        bd = preprocess(*args, **kw)
        state["mt"] = {k: v.clone() for k, v in bd.items()}
        return state.get("replace", lambda b: b)(bd)

    backend.preprocess = hooked
    data = memmap(a.split)

    def energy(s):
        lo, hi = data["na"][s], data["na"][s + 1]
        cell = torch.tensor(data["c"][s], dtype=torch.float64)
        sys_ = System(types=torch.tensor(data["a"][lo:hi]), positions=torch.tensor(data["x"][lo:hi], dtype=torch.float64),
                      cell=cell, pbc=(cell != 0).any(dim=1))
        sys_ = get_system_with_neighbor_lists(sys_, model.requested_neighbor_lists())
        out = model([sys_], {"energy": ModelOutput()})
        return float(out["energy"].block().values.sum())

    for s in a.ids:
        n = int(data["na"][s + 1] - data["na"][s])
        txt = (Path(a.dir) / f"{s}.txt").read_text()
        E_c = float(re.search(r"E engine ([-0-9.]+)\s+cpu ([-0-9.]+)", txt)[2])
        mi = read_inputs(Path(a.dir) / f"{s}.bin")
        state.pop("replace", None)
        E_mt = energy(s)
        mt = state["mt"]

        valid = torch.tensor(mi["padding_mask"] < 0.5)
        v = torch.tensor(mi["edge_vectors"], dtype=torch.float64)
        d = torch.sqrt((v * v).sum(-1) + 1e-15)
        f = torch.where(valid, cutoff_func_bump(d, torch.tensor(mi["cutoff_values"], dtype=torch.float64), width), 0.0)
        mc = dict(element_indices_nodes=torch.tensor(mi["species"]).long(),
                  element_indices_neighbors=torch.tensor(mi["neighbor_species"]).long(),
                  edge_vectors=v, edge_distances=d, padding_mask=valid,
                  reverse_neighbor_index=torch.tensor(mi["reverse"]).long(), cutoff_factors=f)
        state["replace"] = lambda b: {**b, **mc}
        E_swap = energy(s)

        # match mlip.cpp slots to metatrain slots per centre: species, then nearest vector
        mv, mm = mt["edge_vectors"], mt["padding_mask"]
        perm = -np.ones(mm.shape, int)
        dv = df = 0.0
        species_bad = rev_bad = 0
        for i in range(n):
            js = np.flatnonzero(mm[i].numpy())
            ks = np.flatnonzero(valid[i].numpy())
            if len(js) != len(ks):
                print(f"  atom {i}: {len(js)} metatrain edges, {len(ks)} mlip.cpp")
                continue
            for j in js:
                k = ks[np.argmin(((v[i, ks] - mv[i, j]) ** 2).sum(-1))]
                perm[i, j] = k
                dv = max(dv, float((v[i, k] - mv[i, j]).abs().max()))
                df = max(df, abs(float(f[i, k] - mt["cutoff_factors"][i, j])))
                species_bad += int(mc["element_indices_neighbors"][i, k] != mt["element_indices_neighbors"][i, j])
        M_mt = mm.shape[1]
        for i in range(n):  # the reverse of slot (i, j) in mlip.cpp must be the matched reverse of metatrain's
            for j in np.flatnonzero(mm[i].numpy()):
                r = int(mt["reverse_neighbor_index"][i, j])
                rk = int(mc["reverse_neighbor_index"][i, perm[i, j]])
                rev_bad += int(perm[r // M_mt, r % M_mt] != rk % valid.shape[1] or rk // valid.shape[1] != r // M_mt)
        # metatrain's inputs with only the cutoff factors from mlip.cpp (matched)
        fm = mt["cutoff_factors"].clone()
        for i in range(n):
            for j in np.flatnonzero(mm[i].numpy()):
                fm[i, j] = f[i, perm[i, j]]
        state["replace"] = lambda b: {**b, "cutoff_factors": fm}
        E_f = energy(s)
        state.pop("replace")

        meV = lambda E: f"{1e3 * (E - E_mt) / n:+.4f}"
        print(f"\n=== {s} ({n} atoms; M metatrain {M_mt}, mlip.cpp {valid.shape[1]}; edges {int(mm.sum())} / {int(valid.sum())})")
        print(f"  E - metatrain (meV/atom): swap {meV(E_swap)}  mlip.cpp {meV(E_c)}  metatrain+mlip.cpp factors {meV(E_f)}")
        print(f"  inputs: max |dv| {dv:.3g} A, max |d factor| {df:.3g}, neighbour species mismatches {species_bad}, "
              f"reverse mismatches {rev_bad}, species nodes equal {bool((mc['element_indices_nodes'] == mt['element_indices_nodes']).all())}")


if __name__ == "__main__":
    main()
