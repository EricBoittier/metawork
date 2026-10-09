---
tags: [index]
---
# Home

Second brain for COSMO / metatensor work. One note per topic; commands are copy-pasteable.
Add new things to [[Inbox]] first, sort them later.

## Git
- [[Git cheatsheet]] — everyday commands, undo, rebase, stash
- [[Fork and upstream workflow]] — origin vs upstream, PRs with `gh`, the rules
- [[Submodules in metawork]] — syncing the superproject without clobbering work

## Conda / Python envs
- [[Conda feedstock protocol]] — rerender, bump `git_rev`, build locally, PR order
- [[Conda and uv envs]] — miniforge, conda-smithy env, uv venv, torch/CUDA fixes

## HPC
- [[Clusters overview]] — every cluster at a glance, which to pick, GPU arch table
- Sites: [[EPFL SCITAS (Kuma, Lyra, Jed)]] · [[CSCS Alps]] · [[sciCORE]]
- [[Kuma and SLURM]] — Kuma QOS + `hpc_run.py` job specs
- [[SLURM cheatsheet]] — submit/monitor, arrays, dependencies, requeue, pending reasons
- [[Containers on HPC]] — CSCS Container Engine (podman/enroot/EDF), Apptainer
- [[Modules and Spack]] — Lmod, Spack specs, uenv recipes
- [[Storage and data transfer]] — home vs scratch vs project, rsync, quotas
- [[Remote dev on clusters]] — tmux, Jupyter on compute nodes, VS Code Remote-SSH
- [[SSH and tunnels]] — keys, port forwards, cosmopc machines

## Dev
- [[metawork setup]] — `setup-metawork.sh`, repo layout, where things live
- [[Testing, lint and docs]] — tox envs, ruff

## Languages
- [[C]] — compile/link flags, inspecting `.so`s, C API conventions
- [[C++ and CMake]] — CMake workflow, CUDA archs, ABI gotchas, sanitizers
- [[Rust]] — cargo, clippy, FFI/`cbindgen` pattern, ownership table
- [[Python]] — packaging, debugging (py-spy, faulthandler), perf

## GPU
- [[CUDA]] — our GPUs + `sm_` table, nvcc, kernel basics, Nsight/compute-sanitizer
- [[Triton]] — kernel skeleton, autotune, `do_bench`, interpreter debugging

## ML
- Frameworks: [[PyTorch]] · [[PyTorch extensions and TorchScript]] · [[JAX]]
- Distributed: [[Distributed overview]] · [[DDP]] · [[FSDP]] · [[Scaling]]
- [[Benchmarking and profiling]] — rules for trustworthy numbers, profilers, symptom table
- [[Training practice]] — pre-flight checklist, debugging loss, atomistic specifics
- [[Atomistic ML stack]] — how metatensor/metatomic/metatrain/engines fit together

## Templates
- [[Protocol template]] · [[Command snippet template]]
