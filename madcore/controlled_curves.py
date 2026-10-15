"""Results of the controlled subset runs (mlipcpp-subset.sbatch): FPS vs random.

For each run directory (errors.test.bin, online.bin, rows.npy, and its log):
test energy / force RMSE on the uniform 2^24 test split, on its most diverse
part (rows below 2^16 and 2^18: the test splits of the small FPS coresets,
nested in the 2^24 one), on OMat24 test structures, the 90th percentile of
per-structure force RMSE, wall time, atoms seen, and the first epoch's mean
online force error (each structure scored before its update: learning speed).

    python controlled_curves.py RUN_DIR ... [-o curves.json]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # make_split, also under python -I
from make_split import load_split  # noqa: E402

RECORD = np.dtype([("id", "<i8"), ("step", "<i4"), ("n", "<i4"), ("de", "<f4"), ("fse", "<f4")])
LOGS = Path("/home/boittier/data/madcore/sweep/logs")
GROUPS = Path("/work/cosmo/boittier/lyra/madcore-data/groups.npz")


def rmse(r):
    if not len(r):
        return None, None
    return (float(1e3 * np.sqrt(np.mean(r["de"].astype(np.float64) ** 2))),
            float(1e3 * np.sqrt(r["fse"].astype(np.float64).sum() / (3 * r["n"].sum()))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    a = ap.parse_args()
    test_rows = load_split()["test"]
    test_rows = test_rows[test_rows < 2 ** 24]
    g = np.load(GROUPS)
    omat = int(np.flatnonzero(g["groups"] == "OMat24")[0])
    is_omat = g["group"][test_rows] == omat
    out = {}
    for d in a.runs:
        name = d.name.rsplit("-", 1)[0].replace("mlipcpp-subset-", "")
        job = d.name.rsplit("-", 1)[1]
        r = np.fromfile(d / "errors.test.bin", RECORD)
        rows = test_rows[r["id"]]
        res = {"job": job}
        res["all"] = rmse(r)
        res["fps16_test"] = rmse(r[rows < 2 ** 16])
        res["fps18_test"] = rmse(r[rows < 2 ** 18])
        res["omat24_test"] = rmse(r[is_omat[r["id"]]])
        f_struct = 1e3 * np.sqrt(r["fse"] / (3 * r["n"]))
        res["f_p90"] = float(np.percentile(f_struct, 90))
        log = next(LOGS.glob(f"*-{job}.out")).read_text(errors="replace")
        m = re.search(r"structures \((\w+)[^)]*\), ([\d,]+) atoms", log)
        ep = re.search(r"epochs=(\d+)", log)
        el = re.search(r"elapsed (\d+) s", log)
        res["atoms_seen"] = int(m[2].replace(",", "")) * int(ep[1]) if m and ep else None
        res["elapsed_s"] = int(el[1]) if el else None
        res["structures"] = int(len(np.load(d / "rows.npy")))
        on = np.fromfile(d / "online.bin", RECORD)
        res["online_f_mean"] = float(1e3 * np.sqrt(on["fse"].astype(np.float64).sum() / (3 * on["n"].sum()))) if len(on) else None
        out[name] = res
    keys = ["all", "fps16_test", "fps18_test", "omat24_test"]
    print(f"{'run':16s} {'n':>9s}" + "".join(f" {k + ' E/F':>22s}" for k in keys) + f" {'F p90':>7s} {'online F':>9s} {'min':>5s} {'atoms seen':>11s}")
    for n, v in sorted(out.items()):
        print(f"{n:16s} {v['structures']:>9,}" + "".join(
            f" {v[k][0]:>10.1f}/{v[k][1]:>10.1f}" if v[k][0] is not None else f" {'-':>22s}" for k in keys) +
            f" {v['f_p90']:>7.0f} {v['online_f_mean'] or 0:>9.0f} {(v['elapsed_s'] or 0) / 60:>5.0f} {v['atoms_seen'] or 0:>11.3g}")
    if a.output:
        a.output.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
