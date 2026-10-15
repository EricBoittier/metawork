"""Test errors of mlip.cpp checkpoints grouped by MAD-CORE subset.

Reads the per-structure records of `mlipcpp-train --dump-errors` (one file
per model, all on the same split of the same 2^K memmap), maps split entries
back to MAD-CORE rows (entry i = i-th smallest row of the split below 2^K)
and groups by `dataset_group` (or `dataset_id`) from extract_groups.py.
Per group and model: structures, atoms, energy RMSE / MAE (meV/atom) and
force RMSE (meV/A, over components). Optionally drops the rows of an
exclusion file, to compare errors with and without them.

    python subset_errors.py --level 24 --groups groups.npz LABEL=PREFIX.test.bin ... [--by dataset] [-o out.json]
"""

import argparse
import json
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # make_split, also under python -I

from make_split import load_split

RECORD = np.dtype([("id", "<i8"), ("step", "<i4"), ("n", "<i4"), ("de", "<f4"), ("fse", "<f4")])


def read_errors(path):
    return np.fromfile(path, RECORD)


def metrics(r):
    n_atoms = r["n"].sum()
    return {
        "structures": int(len(r)),
        "atoms": int(n_atoms),
        "rmse_e": float(1e3 * np.sqrt(np.mean(r["de"].astype(np.float64) ** 2))) if len(r) else None,
        "mae_e": float(1e3 * np.mean(np.abs(r["de"]))) if len(r) else None,
        "rmse_f": float(1e3 * np.sqrt(r["fse"].astype(np.float64).sum() / (3 * n_atoms))) if n_atoms else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="+", help="LABEL=errors.bin")
    ap.add_argument("--level", type=int, required=True, help="the memmap's 2^K")
    ap.add_argument("--split", default="test")
    ap.add_argument("--groups", required=True, type=Path)
    ap.add_argument("--by", choices=["group", "dataset"], default="group")
    ap.add_argument("--exclude-rows", type=Path, help=".npy of global rows to leave out")
    ap.add_argument("-o", "--output", type=Path)
    a = ap.parse_args()

    rows = load_split()[a.split]
    rows = rows[rows < 2 ** a.level]
    g = np.load(a.groups)
    code = g[a.by][rows]
    names = g[a.by + "s"]
    drop = np.isin(rows, np.load(a.exclude_rows)) if a.exclude_rows else np.zeros(len(rows), bool)

    out = {"level": a.level, "split": a.split, "by": a.by, "models": {}}
    for spec in a.models:
        label, path = spec.split("=", 1)
        r = read_errors(path)
        assert r["id"].max() < len(rows), f"{path}: split index beyond the 2^{a.level} {a.split} split"
        r = r[~drop[r["id"]]]
        c = code[r["id"]]
        per = {str(names[k]): metrics(r[c == k]) for k in np.unique(c)}
        out["models"][label] = {"all": metrics(r), "groups": per}

    labels = list(out["models"])
    first = out["models"][labels[0]]["groups"]
    order = sorted(first, key=lambda k: -first[k]["structures"])
    head = f"{a.by:34s} {'n':>8s}" + "".join(f" {'E ' + l:>14s} {'F ' + l:>14s}" for l in labels)
    print(head)
    for k in ["all"] + order:
        ms = [out["models"][l]["all"] if k == "all" else out["models"][l]["groups"].get(k) for l in labels]
        print(f"{k[:34]:34s} {ms[0]['structures']:>8,}" + "".join(
            f" {m['rmse_e']:>14.1f} {m['rmse_f']:>14.1f}" if m else f" {'-':>14s} {'-':>14s}" for m in ms))
    if a.output:
        a.output.write_text(json.dumps(out))
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
