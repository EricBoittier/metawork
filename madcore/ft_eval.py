#!/usr/bin/env python3
"""Exported metatomic models (.pt) against the labels on identical structures: energy,
conservative (-dE/dx) and direct forces, split into molecules / periodic cells.
Also reports how many structures of each eval split appear in a training split (hash of Z, x, cell).

    python ft_eval.py --split NAME=DIR[:stride] ... --train DIR ... --out report.json model.pt ...
"""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import ase
import numpy as np
import torch
from metatomic_ase import MetatomicCalculator

warnings.filterwarnings("ignore")


def load(d):
    na = np.load(f"{d}/na.npy")
    mm = lambda f, t, *s: np.memmap(f"{d}/{f}", t, "r").reshape(-1, *s)
    return dict(na=na, x=mm("x.bin", np.float32, 3), z=mm("a.bin", np.int32), c=mm("c.bin", np.float32, 3, 3),
                e=mm("energy.bin", np.float32), f=mm("non_conservative_force.bin", np.float32, 3))


def key(s, i):
    lo, hi = s["na"][i], s["na"][i + 1]
    return hashlib.blake2b(s["z"][lo:hi].tobytes() + s["x"][lo:hi].tobytes() + s["c"][i].tobytes(), digest_size=16).digest()


def stats(err):
    err = np.asarray(err)
    return dict(rmse=float(np.sqrt((err ** 2).mean()) * 1e3), mae=float(np.abs(err).mean() * 1e3)) if len(err) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--train", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--refit", action="store_true", help="per-element offset refit of the energy residuals (needs a val20 split)")
    a = ap.parse_args()

    splits = {}
    for spec in a.split:
        name, rest = spec.split("=")
        d, stride = (rest.split(":") + ["1"])[:2]
        s = load(d)
        splits[name] = (s, np.arange(0, len(s["na"]) - 1, int(stride)))
    report = {"overlap": {}, "models": {}}
    for t in a.train:
        tr = load(t)
        keys = {key(tr, i) for i in range(len(tr["na"]) - 1)}
        for name, (s, ids) in splits.items():
            report["overlap"][f"{name} in {t}"] = int(sum(key(s, i) in keys for i in ids))
            print(f"overlap: {report['overlap'][f'{name} in {t}']} / {len(ids)} of {name} in {t}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for pt in a.models:
        cons = MetatomicCalculator(pt, device=dev)
        nc = MetatomicCalculator(pt, device=dev, non_conservative="forces", do_gradients_with_energy=False)
        report["models"][pt] = {}
        for name, (s, ids) in splits.items():
            t0 = time.time()
            err = {k: {"mol": [], "pbc": []} for k in ("E", "F_cons", "F_nc")}
            e_pred = []
            for i in ids:
                lo, hi = s["na"][i], s["na"][i + 1]
                cell = s["c"][i].astype(np.float64)
                pbc = np.any(cell != 0, axis=1)
                at = ase.Atoms(numbers=s["z"][lo:hi], positions=s["x"][lo:hi].astype(np.float64), cell=cell, pbc=pbc)
                kind = "pbc" if pbc.any() else "mol"
                at.calc = cons
                e_pred.append(at.get_potential_energy())
                err["E"][kind].append((e_pred[-1] - s["e"][i]) / len(at))
                err["F_cons"][kind].extend((at.get_forces() - s["f"][lo:hi]).ravel())
                at.calc = nc
                err["F_nc"][kind].extend((at.get_forces() - s["f"][lo:hi]).ravel())
            r = {k: {g: stats(v[g]) for g in ("mol", "pbc")} | {"all": stats(v["mol"] + v["pbc"])} for k, v in err.items()}
            r["n"] = {g: len(err["E"][g]) for g in ("mol", "pbc")}
            report["models"][pt][name] = r
            np.save(f"{Path(pt).stem}.{name}.energy.npy", np.asarray(e_pred))  # per structure, in ids order (eV)
            print(f"{pt} {name} ({len(ids)} structures, {time.time() - t0:.0f} s): "
                  + "  ".join(f"{k} {r[k]['all']['rmse']:.1f}/{r[k]['all']['mae']:.1f}" for k in ("E", "F_cons", "F_nc")),
                  flush=True)
    json.dump(report, open(a.out, "w"), indent=1)
    if a.refit:
        refit(splits, [Path(pt).stem for pt in a.models])


def refit(splits, stems):
    """Energy residuals (pred - label): mean signed error, then one least-squares offset per element
    fitted on the val20 residuals and applied to every split (is the error a composition baseline shift?)."""
    comp = {}
    for name, (s, ids) in splits.items():
        zs = [s["z"][s["na"][i]:s["na"][i + 1]] for i in ids]
        comp[name] = np.stack([np.bincount(z, minlength=119) for z in zs]).astype(np.float64)
    for stem in stems:
        res = {n: np.load(f"{stem}.{n}.energy.npy") - s["e"][ids] for n, (s, ids) in splits.items()}
        A, b = comp["val20"], res["val20"]
        mu, *_ = np.linalg.lstsq(A, b, rcond=None)
        for n in splits:
            nat = comp[n].sum(1)
            pa, fixed = res[n] / nat, (res[n] - comp[n] @ mu) / nat
            print(f"refit {stem} {n}: mean signed {pa.mean() * 1e3:+.1f} meV/atom, RMSE {np.sqrt((pa ** 2).mean()) * 1e3:.1f}"
                  f" -> per-element offsets (fit on val20) {np.sqrt((fixed ** 2).mean()) * 1e3:.1f}, MAE {np.abs(fixed).mean() * 1e3:.1f}")
        top = np.argsort(-np.abs(mu))[:8]
        print(f"refit {stem} largest offsets (eV/atom of element):", ", ".join(f"Z{z} {mu[z]:+.3f}" for z in top))


if __name__ == "__main__":
    main()
