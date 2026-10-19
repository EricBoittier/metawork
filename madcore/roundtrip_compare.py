#!/usr/bin/env python3
"""mlip.cpp-trained PET vs metatrain / metatomic, structure by structure.

    roundtrip_compare.py pick SPLIT -o ids.txt [--n 1000 --ends 10]
    roundtrip_compare.py compare --ckpt start.ckpt --pt model.pt --split SPLIT
        --mlipcpp PREFIX [--dump PREFIX.test.bin] [-o out.json]

pick: the --ends smallest (>= 2 atoms) and largest structures plus evenly
spaced ranks of the size order in between, so every size is covered.

compare: evaluates the checkpoint (metatrain PET with the GGUF's baselines) in
float64 and float32 (energy, direct forces, conservative forces by autograd)
and the exported metatomic model, then reads `mlipcpp-md check --out PREFIX`
(CUDA engine, CPU and CUDA GraphModel) and the trainer's --dump-errors records,
and prints max / RMS differences of every implementation against float64.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

COLS = {"engine": (3, 6), "cpu": (9, 12), "gpu": (15, 18)}  # (direct, conservative) in PREFIX.forces.bin


def memmap(d):
    d = Path(d)
    return dict(na=np.load(d / "na.npy"), x=np.fromfile(d / "x.bin", np.float32).reshape(-1, 3),
                a=np.fromfile(d / "a.bin", np.int32), c=np.fromfile(d / "c.bin", np.float32).reshape(-1, 3, 3),
                e=np.fromfile(d / "energy.bin", np.float32))


def pick(args):
    n = np.diff(np.load(Path(args.split) / "na.npy"))
    order = np.argsort(n, kind="stable")
    order = order[n[order] >= 2]
    mid = order[np.linspace(args.ends, len(order) - args.ends - 1, args.n - 2 * args.ends).round().astype(int)]
    ids = np.unique(np.concatenate([order[: args.ends], mid, order[-args.ends:]]))
    np.savetxt(args.output, ids, fmt="%d")
    print(f"{len(ids)} structures, {n[ids].min()}-{n[ids].max()} atoms ({n[ids].sum()} atoms)")


def torch_eval(model, data, ids, dtype, device, conservative):
    import torch
    from metatomic.torch import ModelOutput, System
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

    model = model.to(dtype=dtype, device=device).eval()
    nl = model.requested_neighbor_lists()
    outs = {"energy": ModelOutput(), "non_conservative_force": ModelOutput(per_atom=True)}
    E, Fn, Fc = [], [], []
    for s in ids:
        lo, hi = data["na"][s], data["na"][s + 1]
        cell = torch.tensor(data["c"][s], dtype=dtype)
        pos = torch.tensor(data["x"][lo:hi], dtype=dtype, requires_grad=conservative)
        sys_ = System(types=torch.tensor(data["a"][lo:hi]), positions=pos, cell=cell, pbc=(cell != 0).any(dim=1))
        sys_ = get_system_with_neighbor_lists(sys_, nl).to(device=device)
        out = model([sys_], outs)
        e = out["energy"].block().values.sum()
        E.append(float(e))
        Fn.append(out["non_conservative_force"].block().values.reshape(-1, 3).detach().cpu().double().numpy())
        if conservative:
            (g,) = torch.autograd.grad(e, pos)
            Fc.append(-g.cpu().double().numpy())
    return np.array(E), np.concatenate(Fn), np.concatenate(Fc) if conservative else None


def metatomic_eval(path, data, ids, device):
    import torch
    from metatomic.torch import ModelEvaluationOptions, ModelOutput, System, load_atomistic_model
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

    model = load_atomistic_model(path).to(device)
    dtype = getattr(torch, model.capabilities().dtype)
    opts = ModelEvaluationOptions(length_unit="angstrom", outputs={
        "energy": ModelOutput(quantity="energy", unit="eV"),
        "non_conservative_force": ModelOutput(quantity="force", unit="eV/Angstrom", per_atom=True)})
    E, Fn = [], []
    for s in ids:
        lo, hi = data["na"][s], data["na"][s + 1]
        cell = torch.tensor(data["c"][s], dtype=dtype)
        sys_ = System(types=torch.tensor(data["a"][lo:hi]), positions=torch.tensor(data["x"][lo:hi], dtype=dtype),
                      cell=cell, pbc=(cell != 0).any(dim=1))
        sys_ = get_system_with_neighbor_lists(sys_, model.requested_neighbor_lists()).to(device=device)
        out = model([sys_], opts, check_consistency=True)
        E.append(float(out["energy"].block().values.sum()))
        Fn.append(out["non_conservative_force"].block().values.reshape(-1, 3).detach().cpu().double().numpy())
    return model.capabilities().dtype, np.array(E), np.concatenate(Fn)


def compare(args):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(Path(args.mlipcpp_src) / "scripts"))
    from export_pytorch.export_pet_full import load_local_model

    data = memmap(args.split)
    csv = np.genfromtxt(args.mlipcpp + ".csv", delimiter=",", names=True)
    ids = csv["index"].astype(int)
    n = csv["n_atoms"].astype(int)
    rows = np.fromfile(args.mlipcpp + ".forces.bin", np.float32).reshape(-1, 7, 3).astype(np.float64)
    assert len(rows) == n.sum(), f"{len(rows)} force rows for {n.sum()} atoms"
    lab_E = data["e"][ids].astype(np.float64)
    lab_F = rows[:, 0]
    nat = np.repeat(n, n)

    model = load_local_model(args.ckpt)
    ref = {}
    for name, dt in [("metatrain f64", torch.float64), ("metatrain f32", torch.float32)]:
        E, Fn, Fc = torch_eval(model, data, ids, dt, args.device, conservative=True)
        ref[name] = dict(E=E, Fn=Fn, Fc=Fc)
        print(f"{name}: done", flush=True)
    dtype, E, Fn = metatomic_eval(args.pt, data, ids, args.device)
    ref[f"metatomic .pt ({dtype})"] = dict(E=E, Fn=Fn, Fc=None)
    for name, c in [("mlip.cpp CUDA engine", "engine"), ("GraphModel CPU", "cpu"), ("GraphModel CUDA", "gpu")]:
        E = csv[{"engine": "E_engine", "cpu": "E_cpu", "gpu": "E_gpu"}[c]]
        if np.isnan(E).all():
            continue
        ref[name] = dict(E=E, Fn=rows[:, COLS[c][0] // 3], Fc=rows[:, COLS[c][1] // 3])
    if args.dump:  # the trainer's own evaluation: energy error per atom and sum |dF|^2 per structure
        rec = np.fromfile(args.dump, dtype=[("id", "<i8"), ("step", "<i4"), ("n", "<i4"), ("de", "<f4"), ("fse", "<f4")])
        by = {int(r["id"]): r for r in rec}
        have = np.array([s in by for s in ids])
        ref["trainer (--dump-errors)"] = dict(
            E=np.where(have, lab_E + np.array([by[s]["de"] if s in by else np.nan for s in ids]) * n, np.nan),
            fse=np.array([by[s]["fse"] if s in by else np.nan for s in ids]))

    base = ref["metatrain f64"]
    diff = lambda a, b: (float(np.sqrt(np.nanmean((a - b) ** 2))), float(np.nanmax(np.abs(a - b))))
    report = {"n_structures": int(len(ids)), "atoms": [int(n.min()), int(n.max()), int(n.sum())], "vs_f64": {}, "vs_labels": {}}
    print(f"\n{len(ids)} structures, {n.min()}-{n.max()} atoms; differences vs metatrain float64 "
          "(rms / max; E in meV/atom, F in meV/A)")
    print(f"{'':28s} {'E':>18s} {'F direct':>20s} {'F conservative':>20s}")
    for name, r in ref.items():
        if r is base:
            continue
        row = {"E": diff(1e3 * r["E"] / n, 1e3 * base["E"] / n)}
        if r.get("Fn") is not None:
            row["Fn"] = diff(1e3 * r["Fn"], 1e3 * base["Fn"])
        if r.get("Fc") is not None:
            row["Fc"] = diff(1e3 * r["Fc"], 1e3 * base["Fc"])
        if "fse" in r:  # per-structure RMS force error, trainer vs float64 direct forces
            fse64 = np.add.reduceat(((base["Fn"] - lab_F) ** 2).sum(1), np.r_[0, np.cumsum(n)[:-1]])
            row["rmsF_per_structure"] = diff(1e3 * np.sqrt(r["fse"] / (3 * n)), 1e3 * np.sqrt(fse64 / (3 * n)))
        report["vs_f64"][name] = row
        fmt = lambda k: f"{row[k][0]:9.3g} / {row[k][1]:8.3g}" if k in row else ""
        print(f"{name:28s} {fmt('E'):>18s} {fmt('Fn') or fmt('rmsF_per_structure'):>20s} {fmt('Fc'):>20s}")
    print("\nvs labels (RMSE / MAE; E meV/atom, F meV/A over components)")
    for name, r in ref.items():
        if "Fn" not in r:
            continue
        e = 1e3 * (r["E"] - lab_E) / n
        f = 1e3 * (r["Fn"] - lab_F)
        row = {"E": [float(np.sqrt(np.nanmean(e ** 2))), float(np.nanmean(np.abs(e)))],
               "Fn": [float(np.sqrt(np.nanmean(f ** 2))), float(np.nanmean(np.abs(f)))]}
        report["vs_labels"][name] = row
        print(f"{name:28s} E {row['E'][0]:8.2f} / {row['E'][1]:8.2f}   F_direct {row['Fn'][0]:8.1f} / {row['Fn'][1]:8.1f}")
    # the worst structures for the CUDA engine vs float64 direct forces
    if "mlip.cpp CUDA engine" in ref:
        d = np.abs(ref["mlip.cpp CUDA engine"]["Fn"] - base["Fn"]).max(1)
        per = np.maximum.reduceat(d, np.r_[0, np.cumsum(n)[:-1]])
        worst = np.argsort(per)[-5:][::-1]
        report["worst_engine_direct"] = [[int(ids[w]), int(n[w]), 1e3 * float(per[w])] for w in worst]
        print("largest engine direct-force differences (index, atoms, meV/A):", report["worst_engine_direct"])
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("split")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--ends", type=int, default=10)
    c = sub.add_parser("compare")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--pt", required=True)
    c.add_argument("--split", required=True)
    c.add_argument("--mlipcpp", required=True, help="mlipcpp-md check --out PREFIX")
    c.add_argument("--mlipcpp-src", default="/home/boittier/metawork/mlip.cpp")
    c.add_argument("--dump", help="trainer PREFIX.test.bin")
    c.add_argument("--device", default="cuda")
    c.add_argument("-o", "--output")
    args = ap.parse_args()
    pick(args) if args.cmd == "pick" else compare(args)


if __name__ == "__main__":
    main()
