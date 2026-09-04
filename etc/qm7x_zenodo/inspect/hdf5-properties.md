# QM7-X HDF5 properties

Source: the `README.txt` on
[Zenodo record 4288677](https://zenodo.org/records/4288677)
([Hoja *et al.*, *Sci. Data* **8**, 43 (2021)](https://www.nature.com/articles/s41597-021-00812-2)).
Layouts below are per conformation group.

**this example** = written by `zenodo_to_metatensor.py` and used by a YAML
in [`../options/`](../options/). Other keys follow the same pattern: pick
a sample kind and a tensor type, wrap a `TensorBlock`, save `.mts`.


## Geometry

| HDF5 key | unit | layout | this example |
| --- | --- | --- | --- |
| `atNUM` | — | `(N,)` | atomic numbers |
| `atXYZ` | Å | `(N, 3)` | positions |
| `sRMSD` | Å | scalar | RMSD to the optimized structure |
| `sMIT` | amu·Å² | `(9,)` | moment of inertia tensor |


## Energies (structure scalars)

| HDF5 key | unit | this example |
| --- | --- | --- |
| `eAT` | eV | `energy` (PBE0 atomization; forces match total energy) |
| `ePBE0+MBD` | eV | total PBE0+MBD |
| `eDFTB+MBD` | eV | total DFTB+MBD |
| `ePBE0` | eV | PBE0 without MBD |
| `eMBD` | eV | MBD dispersion |
| `eTS` | eV | TS dispersion |
| `eNN` | eV | nuclear–nuclear repulsion |
| `eKIN` | eV | kinetic |
| `eNE` | eV | nuclear–electron attraction |
| `eEE` | eV | classical electron–electron Coulomb |
| `eXC` | eV | exchange–correlation |
| `eX` | eV | exchange |
| `eC` | eV | correlation |
| `eXX` | eV | exact exchange |
| `eKSE` | eV | sum of Kohn–Sham eigenvalues |
| `eH` | eV | HOMO |
| `eL` | eV | LUMO |
| `HLgap` | eV | `mtt::hlgap` |


## Electronic structure / multipoles

| HDF5 key | unit | layout | this example |
| --- | --- | --- | --- |
| `KSE` | eV | `(n_occ+n_virt,)` | Kohn–Sham eigenvalues (length depends on the molecule) |
| `DIP` | e·Å | scalar | dipole magnitude |
| `vDIP` | e·Å | `(3,)` | `mtt::dipole` |
| `vTQ` | e·Å² | `(3,)` | total quadrupole components |
| `vIQ` | e·Å² | `(3,)` | ionic quadrupole |
| `vEQ` | e·Å² | `(3,)` | electronic quadrupole |
| `mC6` | Eh·a₀⁶ | scalar | `mtt::c6` (XYZ only; optional) |
| `mPOL` | a₀³ | scalar | SCS molecular polarizability (isotropic) |
| `mTPOL` | a₀³ | `(9,)` / `(3, 3)` | `mtt::polarizability` (spherical λ=0,2 or Cartesian rank 2) |


## Forces and atom-in-molecule quantities

| HDF5 key | unit | layout | this example |
| --- | --- | --- | --- |
| `totFOR` | eV/Å | `(N, 3)` | `energy` positions gradient (PBE0+MBD, unitary-force cleaned) |
| `pbe0FOR` | eV/Å | `(N, 3)` | PBE0 forces |
| `vdwFOR` | eV/Å | `(N, 3)` | MBD forces |
| `hVOL` | a₀³ | `(N,)` | Hirshfeld volumes |
| `hRAT` | — | `(N,)` | Hirshfeld ratios |
| `hCHG` | e | `(N,)` | `mtt::hirshfeld_charge` |
| `hDIP` | e·a₀ | `(N,)` | Hirshfeld dipole magnitudes |
| `hVDIP` | e·a₀ | `(N, 3)` | `mtt::hirshfeld_dipole` |
| `atC6` | Eh·a₀⁶ | `(N,)` | `mtt::atomic_c6` (XYZ only; optional) |
| `atPOL` | a₀³ | `(N,)` | atomic polarizabilities |
| `vdwR` | a₀ | `(N,)` | van der Waals radii |


## TensorMap layouts used here

| target | sample kind | character | how it is stored for metatrain |
| --- | --- | --- | --- |
| `energy` | system | scalar + `positions` gradient | XYZ `info["energy"]` / `arrays["forces"]` |
| `mtt::hlgap` | system | scalar | XYZ `info["hlgap"]` |
| `mtt::dipole` | system | Cartesian rank 1 | XYZ `info["dipole"]` |
| `mtt::hirshfeld_charge` | atom | scalar | XYZ `arrays["hirshfeld_charge"]` |
| `mtt::hirshfeld_dipole` | atom | Cartesian rank 1 | XYZ `arrays["hirshfeld_dipole"]` |
| `mtt::polarizability` | system | spherical λ=0,2 | sidecar `polarizability_spherical.mts` (ASE cannot store two irreps) |
| `mtt::polarizability` | system | Cartesian rank 2 | XYZ `info["polarizability"]` (PET only) |
| `mtt::c6` / `mtt::atomic_c6` | system / atom | scalar | XYZ `info["c6"]` / `arrays["atomic_c6"]` |

`--format zip` writes each of the first six as a `.mts` member of a
`DiskDataset`. `mC6` / `atC6` are left out of the zip on purpose:
DiskDataset zips must be homogeneous and those keys are not in
`REQUIRED_HDF5`.
