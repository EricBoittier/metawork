"""PCA of the MAD-CORE features over every row, streamed from the HDF5 files.

usage: python features_pca.py [DATA_DIR] -o OUT [--components 10] [--workers 20]

Reads every `madcore_features_*.h5` listed in DATA_DIR/index.json, scales the
features as MAD-CORE's selection does, `(features - mean) / scale`, and in two
passes over row ranges (in parallel processes, so the largest file is decoded
on many cores):

  1. accumulates the mean and covariance, in float64, and diagonalises it;
  2. projects every row on the leading `--components` principal components.

Writes to OUT (default `~/data/madcore/pca-full/`):

  pca.npz          mean, eigenvalues (all 512), explained variance ratios,
                   components (512 x k), the feature scaler, and n_rows
  projections.npy  (n_rows, k) float32, row i = global row i = fps_order i
                   (load with np.load(..., mmap_mode="r"))

With --allow-missing it uses only the feature files present (rows of missing
files are NaN in projections.npy), e.g. to try it on the level-18 file.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# one BLAS thread per process: the parallelism is over row ranges
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import h5py  # noqa: E402
import numpy as np  # noqa: E402


CHUNK = 65536  # rows per read, a multiple of the files' (256, 512) chunks
TASK = 1 << 20  # rows per task


def scaler(path):
    with h5py.File(path, "r") as f:
        return f["mean"][:].astype(np.float64), float(f["scale"][()])


def moments(task):
    """Sum and uncentered second moment of the scaled rows of one task."""
    path, start, stop, mean, scale = task
    total, second = np.zeros(mean.shape), np.zeros((len(mean), len(mean)))
    with h5py.File(path, "r") as f:
        features = f["features"]
        for s in range(start, stop, CHUNK):
            z = (features[s : min(s + CHUNK, stop)].astype(np.float64) - mean) / scale
            total += z.sum(0)
            second += z.T @ z
    return total, second, stop - start


def project(task):
    """Project the rows of one task into projections.npy, in place."""
    path, start, stop, offset, mean, scale, center, components, output = task
    out = np.load(output, mmap_mode="r+")
    with h5py.File(path, "r") as f:
        features = f["features"]
        for s in range(start, stop, CHUNK):
            e = min(s + CHUNK, stop)
            z = (features[s:e].astype(np.float64) - mean) / scale
            out[offset + s : offset + e] = ((z - center) @ components).astype(np.float32)
    out.flush()
    return stop - start


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", nargs="?", default=Path("~/data/madcore").expanduser(), type=Path)
    parser.add_argument("-o", "--output", default=Path("~/data/madcore/pca-full").expanduser(), type=Path)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    files = json.loads((args.data / "index.json").read_text())["files"]
    shards = sorted(
        ((m["rows"][0], m["rows"][1], args.data / name) for name, m in files.items() if m["family"] == "features")
    )
    n_rows = max(stop for _, stop, _ in shards)
    missing = [path.name for _, _, path in shards if not path.exists()]
    if missing and not args.allow_missing:
        parser.error(f"missing feature files (download them, or pass --allow-missing): {', '.join(missing)}")
    shards = [s for s in shards if s[2].exists()]

    # every file carries the scaler; they must agree for one scaling
    mean, scale = scaler(shards[0][2])
    for _, _, path in shards[1:]:
        other_mean, other_scale = scaler(path)
        assert np.allclose(other_mean, mean) and np.isclose(other_scale, scale), f"{path}: different scaler"

    tasks = []  # (path, local start, local stop, global offset of the file)
    for first, last, path in shards:
        with h5py.File(path, "r") as f:
            rows = f["features"].shape[0]
        assert rows == last - first, f"{path}: {rows} rows, index.json says {last - first}"
        tasks += [(path, s, min(s + TASK, rows), first) for s in range(0, rows, TASK)]
    used = sum(stop - start for _, start, stop, _ in tasks)
    print(f"{len(shards)} files, {used:,} of {n_rows:,} rows, {len(tasks)} tasks on {args.workers} workers")

    with ProcessPoolExecutor(args.workers) as pool:
        total, second, count = np.zeros(len(mean)), np.zeros((len(mean),) * 2), 0
        for t, s, n in pool.map(moments, [(p, a, b, mean, scale) for p, a, b, _ in tasks]):
            total, second, count = total + t, second + s, count + n
        center = total / count
        covariance = (second - count * np.outer(center, center)) / (count - 1)
        eigenvalues, vectors = np.linalg.eigh(covariance)
        eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
        explained = eigenvalues / eigenvalues.sum()
        components = vectors[:, : args.components]
        print("explained variance:", np.round(explained[: args.components], 4),
              f"(first {args.components}: {explained[: args.components].sum():.3f})")

        args.output.mkdir(parents=True, exist_ok=True)
        output = args.output / "projections.npy"
        projections = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(n_rows, args.components))
        projections[:] = np.nan
        projections.flush()
        del projections
        done = sum(pool.map(project, [(p, a, b, o, mean, scale, center, components, output) for p, a, b, o in tasks]))

    np.savez(
        args.output / "pca.npz",
        center=center, eigenvalues=eigenvalues, explained=explained, components=components,
        feature_mean=mean, feature_scale=scale, n_rows=n_rows, rows_used=used,
    )
    print(f"projected {done:,} rows -> {output}, model -> {args.output / 'pca.npz'}")


if __name__ == "__main__":
    main()
