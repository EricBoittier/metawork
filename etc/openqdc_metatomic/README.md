# OpenQDC → metatensor / metatomic / metatrain

[OpenQDC](https://www.openqdc.io/) is a Python library that downloads and
memory-maps quantum-chemistry datasets into a single array layout (positions,
atomic numbers, energies, forces, isolated-atom baselines). This folder is
about getting those arrays into the metatensor stack:

- **[metatensor](https://docs.metatensor.org/)** stores labelled tensors
  (`TensorMap`) with a shared `.mts` file format across Python, C, C++ and
  Rust.
- **[metatomic](https://docs.metatensor.org/metatomic/)** stores atomistic
  `System` objects (positions, types, cell) as `.mta` files.
- **[metatrain](https://docs.metatensor.org/metatrain/)** trains models from
  XYZ, a zip `DiskDataset`, or a memory-mapped directory.

OpenQDC itself has no C/C++/Rust bindings. The practical pipeline is: load in
Python, convert once, then train with metatrain and/or consume the `.mts`
files from any metatensor language API.

```
OpenQDC memmaps  ──Python──►  TensorMap (.mts) + System (.mta)
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
              metatrain         C / C++ / Rust    metatensor-learn
              (xyz/zip/memmap)  (load .mts)       DataLoader
```

Install OpenQDC next to the rest of this workspace:

```bash
pip install openqdc
# or: conda install -c conda-forge openqdc
```


## QM7-X

[QM7-X](https://www.nature.com/articles/s41597-021-00812-2)
([Hoja *et al.*, *Sci. Data* **8**, 43 (2021)](https://doi.org/10.1038/s41597-021-00812-2),
[Zenodo](https://zenodo.org/records/4288677)) covers ~4.2 million equilibrium
and non-equilibrium structures of small organics (H, C, N, O, S, Cl; up to
seven non-hydrogen atoms). Properties were computed at PBE0+MBD; the original
HDF5 files also store dipoles, polarizabilities, Hirshfeld charges, and so on.

OpenQDC's `QM7X` wrapper keeps the geometries plus:

| `energy_method` | label | forces |
| --- | --- | --- |
| `0` | PBE0+MBD (`ePBE0+MBD`) | yes (`pbe0FOR`) |
| `1` | DFTB3+MBD (`eDFTB+MBD`) | no |

Original units in OpenQDC are eV and Å, which already match the metatrain
examples. The original paper's extra properties (polarizability tensors, …)
are **not** in the OpenQDC energy/force subset; metatrain's test file
`qm7x_reduced_100.xyz` is a small XYZ extract that *does* include
polarizability, used for spherical-target tests.


## Loading from OpenQDC

First access downloads and caches under `~/.cache/openqdc`. After that,
construction only memory-maps the arrays — it does not load 4.2 M structures
into RAM.

```python
from openqdc.datasets import QM7X

dataset = QM7X(
    energy_unit="ev",          # kcal/mol, kj/mol, hartree, ev
    distance_unit="ang",       # ang, nm, bohr
    array_format="torch",      # numpy, torch, jax
    energy_type="formation",   # formation | regression | null
)

print(len(dataset), dataset.energy_methods)
# ~4.2e6, ['pbe0/def2-tzvp', 'dft3b']  (names vary slightly by version)
```

`energy_type="formation"` (the default) subtracts isolated-atom energies so
the label is an atomization / formation energy. That is the usual MLIP
target; `null` leaves total energies and you can let metatrain fit an atomic
baseline instead.

### What a sample looks like

Indexing is the native API. Each item is a dict of arrays, sliced out of the
memmaps:

```python
entry = dataset[0]
# entry["positions"]       (n, 3)
# entry["atomic_numbers"]  (n,)
# entry["charges"]         (n,)
# entry["energies"]        (n_methods,)          — PBE0, DFTB, …
# entry["forces"]          (n, 3, n_force_methods)  — only for methods with forces
# entry["e0"]              (n, n_methods)        — isolated-atom energies
# entry["name"], entry["subset"]
```

For a single ASE object (inspection, visualisation), call `get_ase_atoms`:

```python
atoms = dataset.get_ase_atoms(0, energy_method=0)
```

Do **not** walk the full dataset as ASE objects:

```python
# slow for 4.2 M structures, and the Atoms keys are OpenQDC's, not metatrain's
for atoms in dataset.as_iter(atoms=True):
    ...
```

`as_iter(atoms=True)` / `to_xyz()` build an `ase.Atoms` per frame and store
labels as `atoms.info["energies"]` (an array) and `atoms.info["forces"]`
(shape `(n, 3, n_methods)`). Metatrain reads **`atoms.info["energy"]`**
(a scalar) and **`atoms.arrays["forces"]`** (shape `(n, 3)`). You have to
remap those keys; the helper script below does that.

### Better ways to iterate

A PyTorch `Dataset` is enough for a custom training loop. Use a custom
collate function because molecules have different sizes:

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=list)
for batch in loader:
    positions = [item["positions"] for item in batch]
    energies = [item["energies"][0] for item in batch]  # method 0 = PBE0
```

For a subset (prototyping, tests):

```python
idx = dataset.subsample(n_samples=1000, seed=0)
subset = [dataset[int(i)] for i in idx]
```

For the **full** QM7-X, convert once to a metatrain `MemmapDataset` (see
below) instead of iterating OpenQDC at train time. OpenQDC's on-disk layout
is already concatenated memmaps; metatrain's is the same idea with slightly
different file names (`x.bin` / `a.bin` / `e.bin` vs OpenQDC's
`atomic_inputs` / `energies`).


## Python: OpenQDC arrays → metatensor `TensorMap`

This is the conversion step. A per-structure energy with forces stored as
the gradient of energy w.r.t. positions (`dE/dx = −F`) is what metatrain
and the other language APIs all understand:

```python
import torch
from ase import Atoms
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import systems_to_torch

entry = dataset[0]
numbers = entry["atomic_numbers"]
positions = entry["positions"]
energy = float(entry["energies"][0])          # PBE0+MBD
forces = entry["forces"][:, :, 0]             # (n, 3)

atoms = Atoms(numbers=numbers, positions=positions)
system = systems_to_torch(atoms, dtype=torch.float64)

energy_block = TensorBlock(
    values=torch.tensor([[energy]], dtype=torch.float64),
    samples=Labels("system", torch.tensor([[0]])),
    components=[],
    properties=Labels("energy", torch.tensor([[0]])),
)
energy_block.add_gradient(
    "positions",
    TensorBlock(
        values=-torch.tensor(forces, dtype=torch.float64).unsqueeze(-1),
        samples=Labels(
            names=["sample", "atom"],
            values=torch.tensor([[0, i] for i in range(len(atoms))]),
        ),
        components=[Labels("xyz", torch.tensor([[0], [1], [2]]))],
        properties=Labels("energy", torch.tensor([[0]])),
    ),
)
energy_map = TensorMap(Labels.single(), [energy_block])

energy_map.save("energy.mts")   # also: import metatensor as mts; mts.save(...)
system.save("system.mta")       # metatomic System
```

`metatensor` (numpy backend) and `metatensor.torch` write the same `.mts`
bytes. Use the torch package when the next step is metatrain or a torch
model; use the numpy package when handing files to C/C++/Rust.

For a custom Python training loop that is *not* metatrain, wrap the
`TensorMap`s in [`metatensor.learn.data.Dataset`](https://docs.metatensor.org/latest/examples/learn/index.html):

```python
from metatensor.learn.data import DataLoader, Dataset

learn_ds = Dataset(system=[system], energy=[energy_map])
learn_loader = DataLoader(learn_ds, batch_size=1)
```


## C, C++, Rust: load the same `.mts` file

The `.mts` format is an uncompressed NPZ. Every language binding of
`metatensor-core` reads it. Construct the file in Python (OpenQDC lives
there), then load it wherever the model or analysis runs.

### Python

```python
import metatensor as mts
import metatensor.torch as mts_torch

tensor = mts.load("energy.mts")                 # numpy arrays
tensor = mts_torch.load("energy.mts")           # torch tensors
block = tensor.block()
energy = float(block.values[0, 0])
forces = -block.gradient("positions").values[:, :, 0]
```

### C++

Headers: `<metatensor.hpp>`. `TensorMap::load` allocates
`SimpleDataArray` buffers by default.

```cpp
#include <metatensor.hpp>
#include <iostream>

int main() {
    auto tensor = metatensor::TensorMap::load("energy.mts");
    auto block = tensor.block_by_id(0);
    auto values = block.values();   // NDArray<double>, shape (1, 1)
    std::cout << "energy = " << values(0, 0) << "\n";

    if (block.has_gradient("positions")) {
        auto grad = block.gradient("positions").values();
        // grad(atom, xyz, property); force = -dE/dx
        std::cout << "n_atoms = " << grad.shape()[0] << "\n";
    }

    // build a TensorMap from scratch (e.g. after reading raw OpenQDC dumps)
    auto energy_block = metatensor::TensorBlock(
        std::make_unique<metatensor::SimpleDataArray<double>>(
            metatensor::SimpleDataArray<double>({1, 1}, /*fill=*/ -123.4)
        ),
        metatensor::Labels({"system"}, {{0}}),
        {},
        metatensor::Labels({"energy"}, {{0}})
    );
    std::vector<metatensor::TensorBlock> blocks;
    blocks.push_back(std::move(energy_block));
    auto built = metatensor::TensorMap(
        metatensor::Labels::single(),
        std::move(blocks)
    );
    metatensor::io::save("energy_cpp.mts", built);
}
```

Link against `metatensor` (CMake: `find_package(metatensor)`). Docs:
[C++ TensorMap](https://docs.metatensor.org/latest/core/reference/cxx/tensor.html),
[I/O](https://docs.metatensor.org/latest/core/reference/cxx/io.html).

### Rust

```rust
use metatensor::{Labels, TensorBlock, TensorMap};

fn main() -> Result<(), metatensor::Error> {
    let tensor = TensorMap::load("energy.mts")?;
    let block = tensor.block_by_id(0);
    let values = block.values();
    println!("energy shape = {:?}", values.shape()?);

    // construct and save
    let block = TensorBlock::new(
        vec![-123.4],
        Labels::new(["system"], &[[0]])?,
        Vec::new(),
        Labels::new(["energy"], &[[0]])?,
    )?;
    let built = TensorMap::new(Labels::single()?, vec![block])?;
    built.save("energy_rust.mts")?;
    Ok(())
}
```

`TensorMap::load` / `metatensor::io::load` fill `ndarray` buffers. Crate:
[`metatensor`](https://docs.metatensor.org/latest/core/reference/rust/metatensor/).

### C

The C API is what C++ and Rust wrap. Loading needs a
`mts_create_array_callback_t` that allocates each block's values (the C++
`default_create_array` and the Rust `create_ndarray` are those callbacks).
A minimal load looks like:

```c
#include <metatensor.h>
#include <stdio.h>

/* create_array must allocate an mts_array_t for the given shape/dtype
   and fill `array`. See metatensor.h (mts_create_array_callback_t).
   Implementing the full mts_array_t vtable by hand is verbose — prefer
   the C++ or Rust loaders unless you already have a C array type. */

extern mts_status_t my_create_array(
    const uintptr_t* shape, uintptr_t shape_count,
    DLDataType dtype, mts_array_t* array
);

int main(void) {
    printf("metatensor %s\n", mts_version());

    struct mts_tensormap_t* tensor =
        mts_tensormap_load("energy.mts", my_create_array);
    if (tensor == NULL) {
        fprintf(stderr, "%s\n", mts_last_error());
        return 1;
    }

    const mts_labels_t* keys = NULL;
    mts_tensormap_keys(tensor, &keys);
    printf("n_blocks = %zu\n", keys->count);

    mts_tensormap_free(tensor);
    return 0;
}
```

Docs: [C API I/O](https://docs.metatensor.org/latest/core/reference/c/misc.html).
For new code, C++ or Rust is the less painful way to build `TensorMap`s;
C is the stable ABI those languages (and Python) call.

Systems (positions / types / cell) are **metatomic**, not metatensor-core.
Python and the libtorch C++ API load them with `load_system("system.mta")`.
There is no C/Rust metatomic loader today — keep geometry in XYZ or pass
`.mta` through Python/C++.


## Using the data in metatrain

Metatrain picks a reader from the `read_from` path:

| path | reader | when to use |
| --- | --- | --- |
| `*.xyz` (or any ASE format) | in-memory ASE | subsets, ≲ 10⁴–10⁵ structures |
| `*.zip` | `DiskDataset` | 10⁴–10⁶, random access per structure |
| directory/ | `MemmapDataset` | full QM7-X (~4.2 M), HPC parallel filesystems |

The helper in this folder remaps OpenQDC keys and writes any of the three:

```bash
# 1 000 random PBE0 structures as XYZ (default)
python etc/openqdc_metatomic/openqdc_to_metatrain.py \
    --dataset QM7X --n-samples 1000 --format xyz

# medium DiskDataset
python etc/openqdc_metatomic/openqdc_to_metatrain.py \
    --dataset QM7X --n-samples 10000 --format zip

# full dataset as memmaps (this will take a while, once)
python etc/openqdc_metatomic/openqdc_to_metatrain.py \
    --dataset QM7X --n-samples all --format memmap
```

Then train:

```bash
mtt train etc/openqdc_metatomic/options-qm7x.yaml
```

[`options-qm7x.yaml`](options-qm7x.yaml) is set up for the XYZ output. Adjust
`read_from` and the target `key`s for the other formats:

```yaml
training_set:
  systems:
    read_from: qm7x.xyz          # or qm7x.zip, or qm7x_memmap/
    length_unit: angstrom
  targets:
    energy:
      key: energy                # zip: <i>/energy.mts
                                 # memmap: energy.bin
      unit: eV
      forces:
        key: forces              # xyz only
        # key: energy_forces     # memmap (MemmapWriter name)
                                 # zip: omit — forces are the positions gradient
```

Drop-in from the data-preparation tutorial if you want to write the zip or
memmap yourself: [`01-data_preparation.py`](https://docs.metatensor.org/metatrain/latest/generated_examples/0-beginner/01-data_preparation.html)
and [`dataset formats`](https://docs.metatensor.org/metatrain/latest/concepts/dataset-formats.html).
The writers used by the helper are
`metatrain.utils.data.writers.DiskDatasetWriter` and `MemmapWriter`.

Other OpenQDC classes (`Spice`, `ANI1x`, `QMugs`, `MD22`, …) work the same
way: pass `--dataset Spice`. Keep `energy_method` at `0` unless you want a
second level of theory; only methods with `force_mask[i] == True` get a
forces array.

### Spherical targets (polarizability)

OpenQDC's `QM7X` does not ship the polarizability tensor. The original QM7-X
HDF5 does (`mPOL`), and metatrain already has a 100-structure extract
(`metatrain/tests/resources/qm7x_reduced_100.xyz`) plus
`dump_spherical_targets.py` that converts Cartesian polarizability to an
`.mts` file with `o3_lambda=0,2` blocks. Point a generic target at that
file:

```yaml
targets:
  mtt::polarizability:
    read_from: qm7x_reduced_100.mts
    type:
      spherical:
        irreps:
          - {o3_lambda: 0, o3_sigma: 1}
          - {o3_lambda: 2, o3_sigma: 1}
```

`MemmapDataset` cannot store spherical targets; use XYZ + a side `.mts`, or
a `DiskDataset` zip.


## References

- OpenQDC: <https://www.openqdc.io/> ·
  [docs](https://docs.openqdc.io/) ·
  [paper](https://arxiv.org/abs/2411.19629)
- QM7-X: [Hoja *et al.*, *Sci. Data* **8**, 43 (2021)](https://www.nature.com/articles/s41597-021-00812-2)
- metatensor I/O: [Python](https://docs.metatensor.org/latest/core/reference/python/io.html) ·
  [C](https://docs.metatensor.org/latest/core/reference/c/misc.html) ·
  [C++](https://docs.metatensor.org/latest/core/reference/cxx/io.html) ·
  [Rust](https://docs.metatensor.org/latest/core/reference/rust/metatensor/io/)
- metatrain data: [formats](https://docs.metatensor.org/metatrain/latest/concepts/dataset-formats.html) ·
  [preparation tutorial](https://docs.metatensor.org/metatrain/latest/generated_examples/0-beginner/01-data_preparation.html)
