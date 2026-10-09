---
tags: [hpc, cscs, alps]
---
# CSCS Alps

Docs: https://docs.cscs.ch · metatensor-specific recipes: `~/Documents/metawork/hpc-docs/CSCS-Alps/` (github.com/metatensor/hpc-docs).

## Login (daily)
Signed keys are valid **1 day** (max 5 signings/day). Always via the jump host `ela`.
```bash
cscs-key sign               # browser MFA; writes cert for ~/.ssh/cscs-key
cscs-key sign --headless    # device-code flow when there's no browser
cscs-key sign -d 1min       # custom duration
ssh clariden                # ProxyJump ela is in ~/.ssh/config
ssh -J boittier@ela.cscs.ch boittier@daint.alps.cscs.ch   # without config
scp clariden:/users/boittier/file.tar.gz .
```
Install/update cscs-key: download the `x86_64-unknown-linux-musl` tarball from github.com/eth-cscs/cscs-key releases.

`~/.ssh/config` pattern (all hosts use `IdentityFile ~/.ssh/cscs-key`; add `IdentitiesOnly yes` if ssh offers the wrong key):
```
Host clariden
    HostName clariden.alps.cscs.ch
    ProxyJump ela
    User boittier
    IdentityFile ~/.ssh/cscs-key
    IdentitiesOnly yes
```

## vClusters
| Name | Platform | Nodes |
|---|---|---|
| `daint` | HPC | 4× GH200 (Grace ARM CPU + H100 96 GB) |
| `clariden` | ML | 4× GH200 — pytorch uenv deployed here |
| `santis` | Climate & weather | GH200 |
| `eiger` | HPC (CPU) | AMD EPYC |
| `bristen` | ML | — |
**Grace = aarch64.** x86 conda envs, wheels and binaries won't run.

## Filesystems
| What | Path | Env | Quota | Backup | Cleanup |
|---|---|---|---|---|---|
| Home | `/users/$USER` | `$HOME` | 50 GB, 500k inodes | daily snapshots in `~/.snapshot` (7 d) | none |
| Scratch (daint/eiger) | `/ritom/scratch/cscs/$USER` | `$SCRATCH` | 150 TB, 1M inodes | **none** | untouched 30 d deleted |
| Scratch (santis) | `/capstor/scratch/cscs/$USER` | `$SCRATCH` | same | none | 30 d |
| Scratch (clariden/bristen) | `/iopsstor/scratch/cscs/$USER` | `$SCRATCH` | same | none | **14 d** |
| Store/project | `/capstor/store/<tenant>/<customer>/<group>` | `$STORE` | per project | tape daily | none |
Check: `quota` (on ela or any login node). Builds in `/dev/shm/$USER` (fast, RAM, gone at logout).

## SLURM on Alps
Every job needs `--account=<project>`. Allocate whole GPUs per task:
```bash
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=2 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=64
#SBATCH --time=02:00:00
```
`--ntasks-per-node=4 --gpus-per-task=1` → each rank sees one GPU via `CUDA_VISIBLE_DEVICES`.
Interactive: `srun -A <ACCOUNT> -t 01:00:00 --pty bash`.

## Software: uenv
```bash
uenv image find pytorch                  # what exists
uenv image find pytorch@clariden         # images from another vCluster
uenv image pull pytorch/v2.6.0:v1
uenv start --view=default pytorch/v2.6.0
python -m venv --system-site-packages ./myenv && . myenv/bin/activate
pip install metatrain                    # or pip install -e . for dev
```
In batch: `#SBATCH --uenv=pytorch/v2.6.0:v1` + `#SBATCH --view=default`.
Build our own (LAMMPS + metatomic): `uenv build CSCS-Alps/uenv/lammps-metatomic/ lmp/mta@daint%gh200` from hpc-docs, then `uenv image pull service::lmp/mta:<TAG>@daint%gh200`. Details: `hpc-docs/CSCS-Alps/uenv.md`, and [[Modules and Spack]].

## Distributed training env (uenv — set these manually!)
```bash
export MASTER_ADDR=$(hostname) MASTER_PORT=25678 WORLD_SIZE=$SLURM_NPROCS
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TRITON_HOME=/dev/shm/ OMP_NUM_THREADS=64
export CUDA_CACHE_DISABLE=1              # no JIT cache on shared FS
export MPICH_GPU_SUPPORT_ENABLED=0       # avoids NCCL deadlocks
export NCCL_NET="AWS Libfabric" NCCL_NET_GDR_LEVEL=PHB NCCL_CROSS_NIC=1
export FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=32768
export FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_RX_MATCH_MODE=software FI_MR_CACHE_MONITOR=userfaultfd
srun -c $SLURM_CPUS_PER_TASK --cpu-bind=socket bash -c '
  export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
  . ./myenv/bin/activate
  mtt train options.yaml'
```
Without these, NCCL falls back to slow TCP over Slingshot. Containers with the `aws_ofi_nccl` hook set most of this for you → [[Containers on HPC]].

Related: [[Clusters overview]], [[Distributed overview]], [[SSH and tunnels]]
