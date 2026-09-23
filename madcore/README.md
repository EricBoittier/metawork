# MAD-CORE for training

Everything needed to train a metatrain model on
[MAD-CORE](../notebooks/madcore-dataset-summary.ipynb): a fixed split, a
native converter to metatrain's dataset formats, and an options file that
batches by atom count.

## Split

The record ships **no** train / validation / test split, so
`make_split.py` defines one: 1 % validation and 1 % test, drawn uniformly
over all 2^24 rows with seed 0; the rest is training data. Only the held-out
indices are stored (`splits/madcore-split-seed0.npz`, 0.7 MB, with the
SHA-256 of the `index.json` it was built from); `load_split()` rebuilds
`train` as the complement.

Rows are MAD-CORE's global row index, which is also `fps_order`. Because the
draw is uniform, restricting the split to the first 2^k rows gives a uniform
split of the level-k coreset, so one file serves every level:

| level | train | val | test |
| --- | --- | --- | --- |
| 11 (2,048 rows) | 2,000 | 24 | 24 |
| 18 (262,144 rows) | 256,864 | 2,584 | 2,696 |
| 24 (all rows) | 16,441,672 | 167,772 | 167,772 |

## Converting

`metatomic-core-examples/build/madcore-to-diskdataset` streams the
`.extxyz.gz` shards, checks every structure through the metatomic C API
(`mta_system_create`), and writes one dataset per split:

```bash
cd ~/metawork/metatomic-core-examples && bash build.sh
./build/madcore-to-diskdataset --split ../madcore/splits/madcore-split-seed0.npz \
    -o ~/data/madcore/memmap-2pow18 --max-rows 262144 ~/data/madcore/madcore_*.extxyz.gz
```

`--max-rows` selects a coreset level (omit it for all 2^24 rows). Two formats:

| `--format` | writes | level 18: time / size |
| --- | --- | --- |
| `memmap` (default) | `train/`, `val/`, `test/` MemmapDataset directories (float32 arrays) | 1.9 s / 0.06 GB |
| `zip` | `train.zip`, `val.zip`, `test.zip` DiskDataset archives (per-structure `system.mta` + `.mts`) | 14.5 s / 1.2 GB |

Both include the atom counts `max_atoms_per_batch` needs. Entry `i` of a
split is its `i`-th smallest row: `load_split()["val"][i]` (restricted to
rows below `--max-rows`) maps it back to MAD-CORE.

Memmap stores positions, forces and energies as float32 (metatrain's
format). For MAD-CORE's largest total energies (~10^4 eV) that is ~1 meV of
absolute resolution, well below per-atom training errors.

## Training

```bash
cd ~/data/madcore/memmap-2pow18
mtt train ~/metawork/madcore/options-pet-memmap.yaml
```

`options-pet-memmap.yaml` trains PET on the energy and on the direct
(non-conservative) forces as two targets. The labels are the selection
model's predictions, not DFT. Batches are packed with `max_atoms_per_batch:
1024`, just above the largest structure (1000 atoms), because sizes span 1 to
1000 atoms (mean 12.9) and a fixed `batch_size` would give very uneven
batches.

## What changes for training speed

Measured with metatrain's own dataset classes on the level-18 data:

* **Start-up**: a memmap dataset opens instantly. Reading the same data with
  the ASE reader parses the extxyz at ~3k structures/s, i.e. ~1.5 h and all
  structures held in memory for the full 2^24.
* **Per-structure reads** are the same for memmap and zip, ~2.2k
  structures/s per dataloader worker: the cost is building the `System` and
  `TensorMap`s in Python, not the storage format. Use several dataloader
  workers if the GPU waits on data.
* **Batches** are packed by atom count, so step cost stays even across
  the wide size range instead of alternating between tiny and huge batches.
