"""Where the test error of mlip.cpp checkpoints comes from, on MAD-CORE.

Reads `--dump-errors` output (PREFIX.test.bin per structure and
PREFIX.test.atoms.bin per atom) of models scored on the same 2^K test split
and breaks the error down by: source group, element (per-atom forces),
structure size, periodicity, position in the farthest-point order, the
label's per-atom force magnitude, and the geometry scan's closest-contact
ratio. Per bin and model: count, energy RMSE (meV/atom), force RMSE
(meV/A over components) and the bin's share of the total squared force
error. Writes JSON for the page and prints the tables.

    python error_breakdown.py --level 24 LABEL=PREFIX ... [-o breakdown.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # make_split, also under python -I
from make_split import load_split  # noqa: E402

RECORD = np.dtype([("id", "<i8"), ("step", "<i4"), ("n", "<i4"), ("de", "<f4"), ("fse", "<f4")])
DATA = Path("/work/cosmo/boittier/lyra/madcore-data")
MEMMAP = Path("/home/boittier/data/madcore")


def load_model(prefix, na):
    r = np.fromfile(f"{prefix}.test.bin", RECORD)
    a = np.fromfile(f"{prefix}.test.atoms.bin", np.float32)
    assert len(a) == r["n"].sum(), f"{prefix}: {len(a)} atom errors for {r['n'].sum()} atoms"
    # memmap atom index of every per-atom error, record by record
    starts = na[r["id"]]
    offs = np.concatenate([[0], np.cumsum(r["n"])[:-1]])
    atom = np.repeat(starts - offs, r["n"]) + np.arange(len(a))
    return r, a, atom


def binned(name, labels, struct_bin, atom_bin, models, kind="structure"):
    """Per bin: n, E RMSE (structures in the bin), F RMSE and share of squared F error
    (atoms in the bin: their own atoms for structure bins, atoms of that kind for atom bins)."""
    out = {"bins": labels, "kind": kind, "models": {}}
    for m, (r, a, _) in models.items():
        sb, ab = struct_bin[m], atom_bin[m]
        tot = a.astype(np.float64).sum()
        rows = []
        for k in range(len(labels)):
            s, at = sb == k, ab == k
            n = int(at.sum()) if kind == "atom" else int(s.sum())
            e = float(1e3 * np.sqrt(np.mean(r["de"][s].astype(np.float64) ** 2))) if s.any() and kind == "structure" else None
            f = float(1e3 * np.sqrt(a[at].astype(np.float64).sum() / (3 * at.sum()))) if at.any() else None
            share = float(a[at].astype(np.float64).sum() / tot) if at.any() else 0.0
            rows.append([n, e, f, share])
        out["models"][m] = rows
    return out


def show(name, b):
    labels = list(b["models"])
    print(f"\n== {name}")
    print(f"{'':28s} {'n':>9s}" + "".join(f" {'E ' + l:>13s} {'F ' + l:>13s} {'F share':>8s}" for l in labels))
    for k, lab in enumerate(b["bins"]):
        rows = [b["models"][l][k] for l in labels]
        if not rows[0][0]:
            continue
        print(f"{str(lab)[:28]:28s} {rows[0][0]:>9,}" + "".join(
            f" {(f'{x[1]:.1f}' if x[1] is not None else '-'):>13s} {(f'{x[2]:.1f}' if x[2] is not None else '-'):>13s} {x[3]:>8.1%}" for x in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="+", help="LABEL=PREFIX (PREFIX.test.bin, PREFIX.test.atoms.bin)")
    ap.add_argument("--level", type=int, default=24)
    ap.add_argument("-o", "--output", type=Path)
    a = ap.parse_args()
    test = MEMMAP / f"memmap-2pow{a.level}" / "test"
    na = np.load(test / "na.npy")
    Z = np.memmap(test / "a.bin", np.int32, "r")
    F = np.memmap(test / "non_conservative_force.bin", np.float32, "r").reshape(-1, 3)
    cell = np.memmap(test / "c.bin", np.float32, "r").reshape(-1, 3, 3)
    rows = load_split()["test"]
    rows = rows[rows < 2 ** a.level]
    g = np.load(DATA / "groups.npz")
    geo = np.load(DATA / "geometry" / "test.npz") if a.level == 24 else None
    models = {lab: load_model(p, na) for lab, p in (m.split("=", 1) for m in a.models)}

    res = {}
    # source group: the 16 largest in the test split, the rest as "other"
    code = g["group"][rows]
    top = np.argsort(-np.bincount(code))[:16]
    gl = [str(g["groups"][k]) for k in top] + ["other"]
    gmap = np.full(code.max() + 1, len(top))
    gmap[top] = np.arange(len(top))
    sb = {m: gmap[code[r["id"]]] for m, (r, _, _) in models.items()}
    ab = {m: np.repeat(sb[m], r["n"]) for m, (r, _, _) in models.items()}
    res["group"] = binned("group", gl, sb, ab, models)

    # element: per-atom force error by atomic number (elements with >= 2000 test atoms)
    zs = {m: np.asarray(Z[atom]) for m, (_, _, atom) in models.items()}
    first = next(iter(zs.values()))
    counts = np.bincount(first, minlength=119)
    from ase.data import chemical_symbols
    keep = [z for z in np.argsort(-counts) if counts[z] >= 2000][:40]
    zmap = np.full(119, len(keep))
    zmap[keep] = np.arange(len(keep))
    zl = [chemical_symbols[z] for z in keep] + ["other"]
    res["element"] = binned("element", zl, {m: np.zeros(len(r), int) - 1 for m, (r, _, _) in models.items()},
                            {m: zmap[zs[m]] for m in models}, models, kind="atom")

    # structure size
    edges = [1, 2, 5, 9, 17, 33, 65, 10 ** 6]
    sl = ["1", "2-4", "5-8", "9-16", "17-32", "33-64", "65+"]
    sb = {m: np.searchsorted(edges, r["n"], side="right") - 1 for m, (r, _, _) in models.items()}
    res["size"] = binned("size", sl, sb, {m: np.repeat(sb[m], models[m][0]["n"]) for m in models}, models)

    # periodicity (number of nonzero cell rows)
    per = (np.abs(np.asarray(cell)).sum(2) > 0).sum(1)
    pl = ["0D (molecule / cluster)", "1D", "2D (slab)", "3D (bulk)"]
    sb = {m: per[r["id"]] for m, (r, _, _) in models.items()}
    res["periodicity"] = binned("periodicity", pl, sb, {m: np.repeat(sb[m], models[m][0]["n"]) for m in models}, models)

    # position in the FPS order: which coreset first contains the structure
    fl = ["< 2^14", "2^14-2^16", "2^16-2^18", "2^18-2^20", "2^20-2^22", "2^22-2^24"]
    fe = np.array([0, 2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20, 2 ** 22, 2 ** 24])
    sb = {m: np.searchsorted(fe, rows[r["id"]], side="right") - 1 for m, (r, _, _) in models.items()}
    res["fps_position"] = binned("fps", fl, sb, {m: np.repeat(sb[m], models[m][0]["n"]) for m in models}, models)

    # label force magnitude per atom (eV/A)
    ml = ["< 0.1", "0.1-1", "1-5", "5-20", "20-50", "> 50"]
    me = np.array([0, 0.1, 1, 5, 20, 50, np.inf])
    ab = {}
    for m, (_, _, atom) in models.items():
        mag = np.sqrt((np.asarray(F[atom]) ** 2).sum(1))
        ab[m] = np.searchsorted(me, mag, side="right") - 1
    res["label_force"] = binned("label force", ml, {m: np.zeros(len(models[m][0]), int) - 1 for m in models}, ab, models, kind="atom")

    # closest contact relative to the covalent sum (geometry scan) and absurd C-C
    if geo is not None:
        cl = ["< 0.5 (overlapping)", "0.5-0.6", "0.6-0.7", ">= 0.7 (no short contact)"]
        ce = np.array([0, 0.5, 0.6, 0.7, np.inf])
        sb = {m: np.searchsorted(ce, geo["min_ratio"][r["id"]], side="right") - 1 for m, (r, _, _) in models.items()}
        res["contact"] = binned("contact", cl, sb, {m: np.repeat(sb[m], models[m][0]["n"]) for m in models}, models)
        bl = ["C-C < 1.0 A", "C-C 1.0-1.2 A", "C-C >= 1.2 A", "no C-C pair"]
        cc = geo["min_cc"]
        be = np.where(~np.isfinite(cc), 3, np.searchsorted(np.array([0, 1.0, 1.2, np.inf]), cc, side="right") - 1)
        sb = {m: be[r["id"]] for m, (r, _, _) in models.items()}
        res["cc"] = binned("cc", bl, sb, {m: np.repeat(sb[m], models[m][0]["n"]) for m in models}, models)

    for k, b in res.items():
        show(k, b)
    if a.output:
        a.output.write_text(json.dumps(res))
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
