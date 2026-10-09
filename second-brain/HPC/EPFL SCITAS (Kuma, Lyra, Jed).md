---
tags: [hpc, scitas, epfl, kuma, lyra]
---
# EPFL SCITAS (Kuma, Lyra, Jed)

Docs: https://scitas-doc.epfl.ch · Needs EPFL network or VPN · Pay-per-use accounts.
```bash
kuma    # alias → ssh boittier@kuma.hpc.epfl.ch
lyra    # alias → ssh boittier@lyra.hpc.epfl.ch
ssh boittier@jed.hpc.epfl.ch
```
Passwordless: `ssh-copy-id -i ~/.ssh/id_ed25519.pub boittier@kuma.hpc.epfl.ch`.
"REMOTE HOST IDENTIFICATION HAS CHANGED" → check fingerprint on the docs page, then `ssh-keygen -R kuma.hpc.epfl.ch`.

## Kuma
**There is no default partition — always pass `-p`.**
| Partition | Nodes | Per node | Max CPU per GPU |
|---|---|---|---|
| `h100` | 84 (`kh001-084`) | 4× H100 94 GB, NVLink, FP64 | 16 |
| `l40s` | 20 (`kl001-020`) | 8× L40S 48 GB, FP32 only | 8 |
| `mig12gb` / `mig24gb` | H100 slices | ~1/7 or 2/7 of an H100 | 2 / 5 |

QOS (same idea on Lyra):
| QOS | Time | Limits | Priority |
|---|---|---|---|
| `normal` (default) | 3 d | ≤ 8 nodes | normal |
| `long` | 7 d | ≤ 8 nodes | low |
| `build` | 4 h | 1 node, 16 cores, 0 GPU | high |
| `debug` | 1 h | ≤ 2 nodes | high |

- RAM is fixed at **5900 MB per CPU core** — `--mem` can't raise it. Need more RAM → ask for more cores.
- Login node has no `nvidia-smi`/`nvcc` → compile CUDA in a `debug` job.
- Pricing (from Nov 2024 MOTD): H100 ≈ CHF 0.52/GPU-h, L40S ≈ CHF 0.21/GPU-h. Check usage with the `Sausage` web tool / MOTD.
- 2× 200 Gb InfiniBand HDR per node; `/scratch` is full-flash.

Interactive GPU:
```bash
srun -p h100 -q debug --gres=gpu:1 -c 16 -t 01:00:00 --pty bash
srun -p mig24gb -q debug --gres=gpu:1 -c 5 -t 01:00:00 --pty bash   # cheap & fast to get
```
Batch template:
```bash
#!/bin/bash
#SBATCH -J train
#SBATCH -p h100
#SBATCH -q normal
#SBATCH --nodes=1 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=16
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%x-%j.out
source ~/metawork/.venv/bin/activate
srun python train.py
```
Or let metawork write it: `.venv/bin/python etc/hpc_run.py spec.yaml [--submit]` (`partition:` is required for kuma) — see [[Kuma and SLURM]].

## Lyra (Blackwell, beta)
| Partition | Nodes | Per node | CPU/GPU | RAM/core |
|---|---|---|---|---|
| `rtx6000` | 14 | 8× RTX PRO 6000 Blackwell 96 GB, 128 cores (EPYC 9535) | 16 | 5900 MB |
| `b200` | 8 | 8× B200, 96 cores (Xeon 8568Y+), 1.5 TB RAM | 12 | 15900 MB |
- No default partition. QOS: `normal` 3 d, `long` 7 d, `build`, `debug` 1 h.
- Beta: free until 30 Sep 2026, but jobs may be killed for maintenance → **checkpoint often**.
- Needs torch built for CUDA ≥ 12.8 (`sm_100`/`sm_120`) → `.venv-blackwell` (branch `infra/blackwell-torch-workaround`), not the shared cu121 `.venv`.

## Jed
CPU-only research cluster (also has an academic partition for students). Same SLURM/QOS conventions; use for CPU-heavy preprocessing, DFT, i-PI drivers.

## Software
```bash
module avail / module spider <name>   # SCITAS Spack-based stack
module load gcc cuda python
```
Containers: Apptainer (`apptainer exec --nv image.sif ...`) — see [[Containers on HPC]].
VS Code / OnDemand / FirecREST are supported — see [[Remote dev on clusters]].

Related: [[Clusters overview]], [[SLURM cheatsheet]], [[Storage and data transfer]]
