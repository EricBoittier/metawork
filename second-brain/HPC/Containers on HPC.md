---
tags: [hpc, containers]
---
# Containers on HPC

## CSCS Alps: Container Engine (podman → enroot → EDF)
1. Build (on a compute node: `srun -A <ACCOUNT> -t 01:00:00 --pty bash`), with podman storage in RAM:
   ```toml
   # ~/.config/containers/storage.conf
   [storage]
   driver = "overlay"
   runroot = "/dev/shm/$USER/runroot"
   graphroot = "/dev/shm/$USER/root"
   ```
   ```bash
   podman build -t mta-ipi .
   enroot import -x mount -o $SCRATCH/mta-ipi.sqsh podman://mta-ipi   # rm the old .sqsh first when rebuilding
   ```
2. Describe it in `~/.edf/mta-ipi.toml`:
   ```toml
   image = "/path/to/mta-ipi.sqsh"
   mounts = ["/capstor", "/iopsstor", "${SCRATCH}:${SCRATCH}"]
   workdir = "/workspace"

   [annotations]
   com.hooks.aws_ofi_nccl.enabled = "true"    # NCCL over Slingshot (multi-GPU / multi-node training)
   com.hooks.aws_ofi_nccl.variant = "cuda12"

   [env]
   CUDA_CACHE_DISABLE = "1"
   TORCH_NCCL_ASYNC_ERROR_HANDLING = "1"
   MPICH_GPU_SUPPORT_ENABLED = "0"
   ```
3. Run: `srun --environment=mta-ipi --pty bash` (or a full path to the `.toml`). `--environment` goes on **srun**, not `#SBATCH`.

Base image: Alps Extended Images (NGC PyTorch rebuilt for Alps), e.g.
`FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.01-py3-alps3`.
Plain NGC images ship OpenMPI 4, which doesn't work with the Alps network.

Metatensor stack inside (build torch extensions from source against the container's torch):
```dockerfile
RUN pip install --no-cache-dir ase metatrain ipi \
 && pip install --no-cache-dir --no-build-isolation --no-binary=metatensor-torch metatensor[torch] \
 && pip install --no-cache-dir --no-build-isolation --no-binary=metatomic-torch metatomic[torch] \
 && pip install --no-cache-dir --no-build-isolation --no-binary=vesin-torch vesin[torch]
```
Full walkthroughs: `hpc-docs/CSCS-Alps/molecular-dynamics-with-i-Pi.md`, `metatrain.md`.

## SCITAS / sciCORE: Apptainer (Singularity)
```bash
apptainer pull pytorch.sif docker://nvcr.io/nvidia/pytorch:25.04-py3
apptainer exec --nv -B $PWD,/scratch pytorch.sif python train.py   # --nv = GPU passthrough
apptainer shell --nv pytorch.sif
apptainer build --fakeroot my.sif my.def                            # if fakeroot is allowed
```
- `-B src:dst` bind mounts; `$HOME` is mounted by default.
- Put `.sif` files on scratch/work, not home (they're GBs).
- `APPTAINER_CACHEDIR=/scratch/$USER/.apptainer` to keep cache off home.

## When to use a container vs a venv
- venv/uenv: fastest iteration, editable installs of our repos.
- Container: reproducible runs, odd system deps, sharing an exact env, ARM (GH200) where wheels are scarce.

Related: [[CSCS Alps]], [[Modules and Spack]], [[Conda and uv envs]]
