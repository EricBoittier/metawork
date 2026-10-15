"""Prequential epiplexity estimates of MAD-CORE training runs (mlip.cpp).

Finzi et al. 2026 (arXiv 2601.03220, Eq. 8): with each example seen once,
|P_preq| ~ sum_i [log 1/P_i(Z_i) - log 1/P_M(Z_i)], the area under the online
loss curve above the final loss; the final loss times the data size estimates
the time-bounded entropy. Here the labels come from a deterministic teacher,
so the final loss is what this student cannot learn at its size, and the area
is the structure it had to absorb. The first epoch of a training run is a
single pass, so it is used as the prequential code; the code length of a
label is Gaussian with fixed sigma (only ratios between runs or subsets are
meaningful, not the absolute number of bits).

  logs:   python epiplexity.py logs RUN.out [RUN.out ...]
          area of the logged step losses of epoch 0 above the mean loss of
          its last --tail fraction, in loss units x structures
  errors: python epiplexity.py errors --log LOG.bin --final FINAL.test.bin --level K --groups groups.npz
          per subset: code length of every training structure at first sight
          (mlipcpp-train --log-errors, records before the update) minus the
          subset's mean code length under the final model (its test errors)
"""

import argparse
import json
import re
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # make_split, also under python -I

RECORD = np.dtype([("id", "<i8"), ("step", "<i4"), ("n", "<i4"), ("de", "<f4"), ("fse", "<f4")])
LN2 = np.log(2.0)


def from_logs(a):
    step_re = re.compile(r"^epoch 0 step (\d+)/(\d+): loss ([\d.eE+-]+)")
    setup_re = re.compile(r"(\d+) steps/epoch of (\d+) x (\d+) structures")
    out = {}
    for path in a.runs:
        steps, losses, per_step = [], [], None
        for line in Path(path).read_text(errors="replace").splitlines():
            if m := setup_re.search(line):
                per_step = int(m[2]) * int(m[3])
                steps, losses = [], []  # the canary ran first: keep the last setup's epoch
            elif m := step_re.match(line):
                steps.append(int(m[1])), losses.append(float(m[3]))
        if not steps or per_step is None:
            print(f"{path}: no epoch-0 step losses")
            continue
        s, l = np.array(steps), np.array(losses)
        final = l[s >= (1 - a.tail) * s.max()].mean()
        # trapezoid over the logged steps, from step 0 (first logged loss held)
        x = np.concatenate([[0], s])
        y = np.concatenate([[l[0]], l]) - final
        area = np.trapezoid(np.clip(y, 0, None), x) * per_step
        n = s.max() * per_step
        out[path] = {"structures": int(n), "final_loss": float(final), "area": float(area),
                     "area_per_structure": float(area / n), "entropy_proxy": float(final * n)}
        print(f"{Path(path).name}: {n:,} structures, final epoch-0 loss {final:.4g}, "
              f"area {area:.4g} (loss x structures), {area / n:.4g} per structure, final x n {final * n:.4g}")
    return out


def code_length(r, sigma_e, sigma_f):
    """Gaussian code length in bits of each structure's labels, beyond the constant."""
    e = r["de"].astype(np.float64) ** 2 / (2 * sigma_e ** 2)
    f = r["fse"].astype(np.float64) / (2 * sigma_f ** 2)
    return e / LN2, f / LN2


def from_errors(a):
    from make_split import load_split

    log = np.fromfile(a.log, RECORD)
    first_epoch = log  # --log-errors-epochs 1: every record is a first sight
    final = np.fromfile(a.final, RECORD)
    g = np.load(a.groups)
    train = load_split()["train"]
    train = train[train < 2 ** a.level]
    test = load_split()["test"]
    test = test[test < 2 ** a.level]
    tr_code, te_code = g[a.by][train][first_epoch["id"]], g[a.by][test][final["id"]]
    names = g[a.by + "s"]
    le, lf = code_length(first_epoch, a.sigma_e, a.sigma_f)
    fe, ff = code_length(final, a.sigma_e, a.sigma_f)
    out = {}
    for k in np.unique(tr_code):
        m, t = tr_code == k, te_code == k
        if not t.any():
            continue
        se = (le[m] - fe[t].mean()).sum()
        sf = (lf[m] - ff[t].mean()).sum()
        out[str(names[k])] = {"train_structures": int(m.sum()), "S_energy_bits": float(se), "S_force_bits": float(sf),
                              "S_bits": float(se + sf), "S_bits_per_structure": float((se + sf) / m.sum()),
                              "H_bits": float((fe[t].mean() + ff[t].mean()) * m.sum())}
    order = sorted(out, key=lambda k: -out[k]["S_bits"])
    total = sum(v["S_bits"] for v in out.values())
    print(f"{a.by:34s} {'train':>9s} {'S (Mbit)':>10s} {'share':>7s} {'S/struct':>10s} {'H (Mbit)':>10s}")
    for k in order:
        v = out[k]
        print(f"{k[:34]:34s} {v['train_structures']:>9,} {v['S_bits'] / 1e6:>10.3f} {v['S_bits'] / total:>7.1%} "
              f"{v['S_bits_per_structure']:>10.1f} {v['H_bits'] / 1e6:>10.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("logs")
    p.add_argument("runs", nargs="+")
    p.add_argument("--tail", type=float, default=0.05, help="fraction of epoch 0 whose mean is the final loss")
    p = sub.add_parser("errors")
    p.add_argument("--log", required=True)
    p.add_argument("--final", required=True, help="--dump-errors test records of the final model")
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--groups", required=True)
    p.add_argument("--by", choices=["group", "dataset"], default="group")
    p.add_argument("--sigma-e", type=float, default=0.01, help="eV/atom")
    p.add_argument("--sigma-f", type=float, default=0.1, help="eV/A")
    for q in sub.choices.values():
        q.add_argument("-o", "--output", type=Path)
    a = ap.parse_args()
    out = from_logs(a) if a.mode == "logs" else from_errors(a)
    if a.output:
        a.output.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
