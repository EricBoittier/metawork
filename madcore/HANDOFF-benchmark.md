# Handoff: benchmarking training on MAD-CORE

For a session that benchmarks metatrain training. Everything below is on
`EricBoittier/metawork`, branch `bench/medium-hardware` (commit `fd04f5e` or
later); `main` has the notebook and the metatomic-core examples but not yet
the split / converter.

## Git rules

* Push **only** to the user's forks (`origin` = `EricBoittier/...`), never to
  `metatensor/*` (the `upstream` remotes of the metatomic / metatensor /
  metatrain checkouts). No PRs or comments upstream.
* The `metatomic` checkout is a detached HEAD with local edits: build from it,
  do not commit to it.

## What exists

| path | what |
| --- | --- |
| `notebooks/madcore-dataset-summary.{ipynb,pdf}` | dataset analysis (2^24 structures, 102 elements, sizes 1-1000 atoms, mean 12.9) |
| `madcore/splits/madcore-split-seed0.npz` | fixed split: 1 % val, 1 % test, uniform; `madcore/make_split.py:load_split()` |
| `metatomic-core-examples/build/madcore-to-diskdataset` | Rust converter: extxyz.gz -> MemmapDataset (default) or DiskDataset zip, per split |
| `madcore/options-pet-memmap.yaml` | PET on energy + non-conservative forces, `max_atoms_per_batch: 1024` |
| `madcore/README.md` | details and measured numbers |

## Setup on a new machine

```bash
# data (the record is a draft: the user provides the preview token)
MC_TOKEN=<token> bash ~/metawork/etc/download-madcore-extxyz.sh --keep-gz   # ~11 GB, keep the .gz
# also fetch index.json and dataset_metadata.csv from the same record if needed

# build (needs cargo, zlib, HDF5 serial headers, C/C++ compilers)
cd ~/metawork/metatomic-core-examples && bash build.sh

# convert, e.g. the level-18 coreset (first 2^18 rows; drop --max-rows for all)
./build/madcore-to-diskdataset --split ../madcore/splits/madcore-split-seed0.npz \
    -o ~/data/madcore/memmap-2pow18 --max-rows 262144 ~/data/madcore/madcore_*.extxyz.gz

# train
cd ~/data/madcore/memmap-2pow18 && mtt train ~/metawork/madcore/options-pet-memmap.yaml
```

The download script decompresses the shards unless `--keep-gz` is given; the
converter reads both `.extxyz` and `.extxyz.gz`.

## Measured so far (workstation, level-18 coreset)

* conversion: memmap 1.9 s / 0.06 GB, zip 14.5 s / 1.2 GB (262,144 structures);
  all 2^24 structures scan in ~100 s with the Rust reader vs ~19 min in Python.
* per-structure reads in metatrain: memmap ~= zip ~= 2.2k structures/s per
  dataloader worker (the cost is building System/TensorMap in Python).
* no training run has been done with these files yet.

## Suggested benchmarks

1. data-loading bound or GPU bound: steps/s vs `num_workers` for memmap;
2. `max_atoms_per_batch` (512 / 1024 / 2048) vs a fixed `batch_size`,
   in structures/s and atoms/s, and step-time variance;
3. memmap vs zip vs the ASE reader: start-up time and steps/s;
4. scaling with coreset level (2^16, 2^18, 2^20) using the same split;
5. time spent fitting the composition model before the first epoch.

## Known issues

* `.mta` written by metatomic-core uses `'<b1'` for `pbc`, which
  metatomic-torch cannot read; the converter works around it
  (`fix_one_byte_descr`). Other `.mta` from the C / C++ examples still have it.
* metatrain's docs say `ns.npy` has shape `(1,)`; the reader needs a 0-d array
  (what the converter writes).
* labels are model predictions (`ecumetric_*`), not DFT; the forces are
  direct, so they are a separate `non_conservative_force` target.
