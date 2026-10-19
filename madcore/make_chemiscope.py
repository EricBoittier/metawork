#!/usr/bin/env python3
"""A chemiscope file of the round-trip evaluation (roundtrip_compare.py --save).

    uv run --no-project --with chemiscope --with ase python make_chemiscope.py \
        predictions.npz -o big24-eval.json.gz [--split SPLIT]

One frame per evaluated test structure, with its MAD-CORE group and dataset.
Structure properties (meV/atom, meV/A): energy errors vs the labels
(mlip.cpp engine, metatrain f64), mlip.cpp - metatrain f64, trainer - f64,
force RMSE. Atom properties: |F label|, |F error| (mlip.cpp direct forces),
|mlip.cpp - metatrain f64|. Arrows: the direct-force error per atom.
Open at https://chemiscope.org (drag the file in) or with chemiscope.show_input.
"""
import argparse
import sys
from pathlib import Path

import ase
import chemiscope
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_split import load_split  # noqa: E402

GROUPS = "/work/cosmo/boittier/lyra/madcore-data/groups.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--split", default="/home/boittier/data/madcore/memmap-2pow24/test")
    ap.add_argument("--level", type=int, default=24)
    a = ap.parse_args()

    p = np.load(a.predictions)
    d = Path(a.split)
    na = np.load(d / "na.npy")
    x = np.fromfile(d / "x.bin", np.float32).reshape(-1, 3)
    Z = np.fromfile(d / "a.bin", np.int32)
    c = np.fromfile(d / "c.bin", np.float32).reshape(-1, 3, 3)
    ids, n = p["ids"], p["n_atoms"]
    rows = load_split()["test"]
    rows = rows[rows < 2 ** a.level][ids]
    g = np.load(GROUPS)

    eng, f64 = "mlipcpp_CUDA_engine", "metatrain_f64"
    dF_lab = p[f"Fn:{eng}"] - p["F_label"]
    dF_ref = p[f"Fn:{eng}"] - p[f"Fn:{f64}"]
    starts = np.r_[0, np.cumsum(n)[:-1]]
    frames = []
    for k, s in enumerate(ids):
        lo, hi = na[s], na[s + 1]
        pbc = (c[s] != 0).any(axis=1)
        at = ase.Atoms(numbers=Z[lo:hi], positions=x[lo:hi], cell=c[s], pbc=pbc)
        at.arrays["force_error"] = dF_lab[starts[k]: starts[k] + n[k]]
        frames.append(at)

    per = lambda E: 1e3 * E / n  # eV -> meV/atom
    norm = lambda v: 1e3 * np.linalg.norm(v, axis=1)
    rms_struct = lambda v: 1e3 * np.sqrt(np.add.reduceat((v ** 2).sum(1), starts) / (3 * n))
    props = {
        "index": dict(target="structure", values=ids, description="row of the 2^24 test memmap"),
        "n_atoms": dict(target="structure", values=n),
        "periodic": dict(target="structure", values=[str(bool((c[s] != 0).any())) for s in ids]),
        "group": dict(target="structure", values=[str(g["groups"][g["group"][r]]) for r in rows]),
        "dataset": dict(target="structure", values=[str(g["datasets"][g["dataset"][r]]) for r in rows]),
        "E label": dict(target="structure", values=per(p["E_label"]), units="meV/atom"),
        "E error mlip.cpp": dict(target="structure", values=per(p[f"E:{eng}"] - p["E_label"]), units="meV/atom"),
        "E error metatrain f64": dict(target="structure", values=per(p[f"E:{f64}"] - p["E_label"]), units="meV/atom"),
        "E mlip.cpp - f64": dict(target="structure", values=per(p[f"E:{eng}"] - p[f"E:{f64}"]), units="meV/atom"),
        "F RMSE mlip.cpp (direct)": dict(target="structure", values=rms_struct(dF_lab), units="meV/A"),
        "max |F mlip.cpp - f64|": dict(target="structure", values=np.maximum.reduceat(norm(dF_ref), starts), units="meV/A"),
        "|F label|": dict(target="atom", values=norm(p["F_label"]), units="meV/A"),
        "|F error| mlip.cpp (direct)": dict(target="atom", values=norm(dF_lab), units="meV/A"),
        "|F mlip.cpp - f64| (direct)": dict(target="atom", values=norm(dF_ref), units="meV/A"),
    }
    if "E:trainer" in p.files:
        props["E trainer - f64"] = dict(target="structure", values=per(p["E:trainer"] - p[f"E:{f64}"]), units="meV/atom")
    if f"Fc:{f64}" in p.files:
        props["|F cons error| f64"] = dict(target="atom", values=norm(p[f"Fc:{f64}"] - p["F_label"]), units="meV/A")

    chemiscope.write_input(
        a.output, structures=frames, properties=props,
        environments=chemiscope.all_atomic_environments(frames),
        shapes={"force_error": chemiscope.ase_vectors_to_arrows(frames, "force_error", target="atom", scale=1.0)},
        metadata=dict(name="big24 145880: mlip.cpp vs metatrain on the 2^24 test split",
                  description="1000 structures of every size; roundtrip-check 175691"),
        settings={"target": "structure", "map": {"x": {"property": "n_atoms", "scale": "log"}, "y": {"property": "E error mlip.cpp"},
                          "color": {"property": "F RMSE mlip.cpp (direct)"}},
                  "structure": [{"color": {"property": "|F error| mlip.cpp (direct)"}}]},
    )
    print(f"wrote {a.output}: {len(frames)} structures, {n.sum()} atoms")


if __name__ == "__main__":
    main()
