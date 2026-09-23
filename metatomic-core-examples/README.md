# metatomic-core examples: reading datasets from C, C++ and Rust

Small programs that use the **metatomic-core C API** (`metatomic.h` /
`metatomic.hpp`, built from `../metatomic/metatomic-core`) together with
metatensor to read atomistic datasets in formats metatomic itself does not
parse: gzip-compressed extended XYZ and HDF5. They were written against the
[MAD-CORE](../notebooks/madcore-dataset-summary.ipynb) record (16.8 M
structures in 11 GB of `.extxyz.gz`, plus 512-dimensional features in `.h5`)
but only the key names are MAD-specific.

| program | language | reads | produces |
| --- | --- | --- | --- |
| `extxyz-gz-to-mta` | C | `.extxyz[.gz]` (zlib) | `mta_system_t` per frame, optionally saved as `.mta` |
| `h5-features-to-mts` | C++ | `.h5` (HDF5 C API) | features as a `.mts` TensorMap; features attached to a `.mta` system as custom data |
| `madcore-scan` | Rust | `.extxyz[.gz]` (flate2 / zlib-rs), one thread per file | per-structure summary and sparse composition as `.mts` TensorMaps; validates every frame as an `mta_system_t` |

## Build

```bash
bash build.sh        # builds metatomic-core with cargo, then the three examples into build/
```

`build.sh` expects the metatomic checkout next to this directory (override
with `METATOMIC_DIR`), zlib, HDF5 (serial, `/usr/include/hdf5/serial`), a C11 /
C++17 compiler and cargo. Two things the CMake build normally takes care of
are done by hand:

* `metatomic/version.h` is generated from `metatomic-core/Cargo.toml`;
* the single-header nlohmann/json (needed by `metatomic.hpp`) is downloaded
  into `build/include` if it is not installed system-wide.

The binaries are linked with an RPATH (not RUNPATH) to the cargo output, so
`libmetatomic.so` also finds the `libmetatensor.so` it depends on.

## Usage

```bash
# C: stream a shard, save every 1000th frame as a metatomic system
./build/extxyz-gz-to-mta ~/data/madcore/madcore_0-2pow18.extxyz.gz --every 1000 --save frames/

# C++: scaled features of the first 2048 rows as a TensorMap, and attach
# row 0's features to the system saved above as custom data "madcore::features"
./build/h5-features-to-mts ~/data/madcore/madcore_features_0-2pow18.h5 head-features.mts \
    --rows 0:2048 --attach frames/0.mta 0 frames/0-with-features.mta

# Rust: whole-dataset summary (+ composition), creating an mta_system_t per frame
./build/madcore-scan -o summary.mts --composition composition.mts --systems \
    ~/data/madcore/madcore_*.extxyz.gz
```

Reading the outputs from Python:

```python
import metatensor
import metatomic.torch as mta

summary = metatensor.load("summary.mts")
columns = summary.info()["quantities"].split(",")        # n_atoms, energy, ...
groups = summary.info()["dataset_group"].split(",")      # decodes the dataset_group column
values = summary.block().values                          # (n_structures, len(columns))

system = mta.load_system("frames/0.mta")
```

## Performance on MAD-CORE

On one workstation (24 cores, data on local disk), built against metatomic `3defa9d9`:

| task | time |
| --- | --- |
| `extxyz-gz-to-mta`, first shard (262,144 frames), single thread | 1.8 s (148k frames/s) |
| `madcore-scan --systems`, all 16,777,216 frames, 7 threads | 102 s, 4 GB RSS |
| the notebook's Python reader (7 processes) for the same pass | ~19 min |

The Rust scan is bounded by the largest shard (8.4 M frames, 6 GB
compressed), since gzip decompression is sequential within a file.

## Notes on the C API

* `mta_system_create` takes ownership of the four DLPack tensors. For a
  streaming reader the tensors must own their buffers (the C example frees
  them in the DLPack deleter), because the system outlives the parser's
  scratch space.
* metatomic requires a zero cell vector along non-periodic directions; extxyz
  files often keep a `Lattice` for molecules, so the readers zero it.
* Custom data names must be `<namespace>::<name>` and match the system dtype
  (float64 here).
* `metatomic-core` is built as `cdylib`/`staticlib` only, so the Rust example
  goes through the C API over FFI (with `dlpk` to build DLPack tensors) rather
  than depending on the crate directly.

`scripts/check_mta.py <input.extxyz.gz> <dir>` compares `.mta` files written
by `extxyz-gz-to-mta` with ASE's reading of the same frames, for downstream
testing.
