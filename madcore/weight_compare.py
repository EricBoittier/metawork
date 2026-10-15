"""Compare the weight distributions of trained mlip.cpp PET checkpoints (GGUF).

Per tensor and model: RMS, largest |w|, excess kurtosis, and the relative
distance travelled from the initial weights |W - W0| / |W0|; for matrices
(both sides > 1) the singular values: largest, stable rank |W|_F^2 / s_max^2
and entropy effective rank exp(H(s / sum s)). Prints a per-kind summary and
writes everything (with pooled histograms of w / RMS and the normalised
spectra of every matrix) as JSON.

    python weight_compare.py --init INIT.gguf LABEL=MODEL.gguf ... [-o weights.json]
"""

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np

GGML_F32, GGML_F16 = 0, 1


def read_gguf(path):
    """{name: float32 array (PyTorch order)} of the F32 / F16 tensors of a GGUF file."""
    data = Path(path).read_bytes()
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def string():
        nonlocal pos
        n = take("Q")
        s = data[pos:pos + n].decode()
        pos += n
        return s

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = take("I"), take("Q")
            return [value(et) for _ in range(n)]
        return take(scalar[t])

    assert data[:4] == b"GGUF", path
    pos = 4
    version, n_tensors, n_kv = take("I"), take("Q"), take("Q")
    assert version >= 2, version
    kv = {}
    for _ in range(n_kv):
        k = string()
        kv[k] = value(take("I"))
    infos = []
    for _ in range(n_tensors):
        name, nd = string(), take("I")
        dims = [take("Q") for _ in range(nd)]
        infos.append((name, dims, take("I"), take("Q")))
    align = kv.get("general.alignment", 32)
    base = (pos + align - 1) // align * align
    out = {}
    for name, dims, t, off in infos:
        if t not in (GGML_F32, GGML_F16):
            continue
        n = int(np.prod(dims))
        dt = np.float32 if t == GGML_F32 else np.float16
        a = np.frombuffer(data, dt, n, base + off).astype(np.float32)
        out[name] = a.reshape(dims[::-1])  # ggml ne is the reversed PyTorch shape
    return out


def kind(name):
    """Tensor family: the name with layer numbers removed."""
    return re.sub(r"\.\d+(?=\.|$)", ".N", name)


def stats(w, w0):
    r = {"shape": list(w.shape), "rms": float(np.sqrt(np.mean(w * w))), "max": float(np.abs(w).max())}
    c = w - w.mean()
    var = np.mean(c * c)
    r["kurtosis"] = float(np.mean(c ** 4) / var ** 2 - 3) if var > 0 else 0.0
    if w0 is not None and w0.shape == w.shape and np.linalg.norm(w0) > 0:
        r["moved"] = float(np.linalg.norm(w - w0) / np.linalg.norm(w0))
    m = w.reshape(w.shape[0], -1) if w.ndim >= 2 else None
    if m is not None and min(m.shape) > 1:
        s = np.linalg.svd(m.astype(np.float64), compute_uv=False)
        p = s / s.sum()
        r.update(s_max=float(s[0]), stable_rank=float((s ** 2).sum() / s[0] ** 2),
                 eff_rank=float(np.exp(-(p * np.log(p + 1e-300)).sum())), rank_max=int(min(m.shape)),
                 spectrum=(s / s[0]).round(5).tolist())
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="+", help="LABEL=path.gguf")
    ap.add_argument("--init", help="initial weights (for |W - W0| / |W0|)")
    ap.add_argument("-o", "--output", default="weights.json")
    a = ap.parse_args()
    init = read_gguf(a.init) if a.init else {}
    models = dict(m.split("=", 1) for m in a.models)
    res = {"init": a.init, "models": {}}
    edges = np.linspace(-6, 6, 121)
    for label, path in models.items():
        W = read_gguf(path)
        per = {n: stats(w, init.get(n)) for n, w in W.items() if w.size > 1}
        pooled = np.concatenate([(w / max(per[n]["rms"], 1e-30)).ravel() for n, w in W.items() if n in per and w.ndim >= 2])
        hist = np.histogram(np.clip(pooled, edges[0], edges[-1]), edges)[0]
        res["models"][label] = {"path": path, "tensors": per, "hist_edges": edges.tolist(), "hist": hist.tolist()}

    # per-kind medians, side by side
    labels = list(models)
    kinds = sorted({kind(n) for m in res["models"].values() for n in m["tensors"]})
    cols = ["rms", "kurtosis", "moved", "stable_rank", "eff_rank"]
    print(f"{'tensor kind':48s}" + "".join(f"{c + ' ' + l:>22s}" for c in cols for l in labels))
    summary = {}
    for k in kinds:
        row = {}
        for l in labels:
            ts = [t for n, t in res["models"][l]["tensors"].items() if kind(n) == k]
            row[l] = {c: float(np.median([t[c] for t in ts if c in t])) if any(c in t for t in ts) else None for c in cols}
        summary[k] = row
        print(f"{k[:48]:48s}" + "".join(
            f"{row[l][c]:>22.3f}" if row[l][c] is not None else f"{'-':>22s}" for c in cols for l in labels))
    res["summary"] = summary
    Path(a.output).write_text(json.dumps(res))
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
