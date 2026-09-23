"""Make a reproducible train / validation / test split of MAD-CORE.

The MAD-CORE record ships no split. This draws the validation and test sets
uniformly at random over all 2^24 rows (rows are identified by their global
index, which equals `fps_order`); everything else is training data. Because
the draw is uniform, the split restricted to the first 2^k rows is also a
uniform split of the level-k coreset, so the same file serves every level:

    train_k = train[train < 2**k]   (and likewise for val / test)

Only the (small) validation and test index arrays are stored; `train` is
their complement and is rebuilt by `load_split`.

usage: python make_split.py [--val 0.01] [--test 0.01] [--seed 0] [--index ~/data/madcore/index.json]
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load_split(path=HERE / "splits" / "madcore-split-seed0.npz"):
    """Return ``{"train": ..., "val": ..., "test": ...}`` sorted int64 row indices."""
    data = np.load(path)
    n_rows = int(data["n_rows"])
    held_out = np.zeros(n_rows, dtype=bool)
    held_out[data["val"]] = held_out[data["test"]] = True
    return {"train": np.flatnonzero(~held_out), "val": data["val"].astype(np.int64), "test": data["test"].astype(np.int64)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val", type=float, default=0.01)
    parser.add_argument("--test", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index", type=Path, default=Path("~/data/madcore/index.json").expanduser())
    args = parser.parse_args()

    index_bytes = args.index.read_bytes()
    files = json.loads(index_bytes)["files"]
    n_rows = sum(m["rows"][1] - m["rows"][0] for m in files.values() if m["family"] == "xyz")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_rows)
    n_val, n_test = round(args.val * n_rows), round(args.test * n_rows)
    val, test = np.sort(order[:n_val]).astype(np.int32), np.sort(order[n_val : n_val + n_test]).astype(np.int32)

    out = HERE / "splits" / f"madcore-split-seed{args.seed}.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        out,
        val=val,
        test=test,
        n_rows=n_rows,
        seed=args.seed,
        index_json_sha256=hashlib.sha256(index_bytes).hexdigest(),
    )
    print(f"{n_rows:,} rows: {n_rows - n_val - n_test:,} train, {n_val:,} val, {n_test:,} test -> {out}")
    for k in (11, 18, 24):
        split = {name: rows[rows < 2**k] for name, rows in load_split(out).items()}
        print(f"  level {k:2d}: " + ", ".join(f"{name} {len(rows):,}" for name, rows in split.items()))


if __name__ == "__main__":
    main()
