---
tags: [hpc, modules, spack]
---
# Modules and Spack

## Environment modules (Lmod)
```bash
module avail                   # what's loadable now
module spider cuda             # search everything incl. hidden by hierarchy
module spider cuda/12.4        # what must be loaded first
module load gcc cuda python    # ml gcc cuda python  (short form)
module list
module purge                   # clean slate — start job scripts with this
module show cuda               # what env vars it sets
module save mystack / module restore mystack   # named collections
```
Hierarchy: compiler → MPI → libraries. If `module load hdf5` fails, load the compiler/MPI first (`spider` tells you).
sciCORE uses EasyBuild names (`Python/3.11.5-GCCcore-13.2.0`); SCITAS uses a Spack-generated stack.

### uv venv on top of modules (pattern that works everywhere)
```bash
module purge && module load gcc cuda python
uv venv --python $(which python) .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```
Record the module list in your job script so runs are reproducible.

## Spack (what uenv and SCITAS stacks are built with)
```bash
git clone --depth=2 https://github.com/spack/spack.git && . spack/share/spack/setup-env.sh
spack find                     # installed
spack info lammps              # versions + variants
spack spec lammps +metatomic   # concretise, see full dependency tree before building
spack install lammps +kokkos +cuda cuda_arch=90
spack load lammps
spack env create myenv && spack env activate myenv && spack add ... && spack install
```
Spec syntax: `pkg@version %compiler +variant ~variant key=value ^dependency@ver`.

## uenv recipes (Alps)
A uenv = Spack environment packaged as a squashfs. Recipe dir has `config.yaml` (spack/spack-packages commit), `compilers.yaml`, `environments.yaml` (packages + variants), `repo/packages/<pkg>/package.py` (custom packages).
Update a package version: bump in `environments.yaml` (+ `spack-packages` commit in `config.yaml` if the version is newer than it).
```bash
uenv build CSCS-Alps/uenv/lammps-metatomic/ lmp/mta@daint%gh200          # CSCS build service
stack-config --build /dev/shm/$USER/lmp-mta --recipe CSCS-Alps/uenv/lammps-metatomic \
  -S ../alps-cluster-config/daint                                          # local, stackinator
```

Related: [[CSCS Alps]], [[Containers on HPC]], [[Conda and uv envs]]
