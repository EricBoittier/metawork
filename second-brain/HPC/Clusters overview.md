---
tags: [hpc, index]
---
# Clusters overview

| Cluster | Site | Login | GPUs | Use it for |
|---|---|---|---|---|
| **Kuma** | EPFL SCITAS | `kuma` → `kuma.hpc.epfl.ch` | H100 94 GB ×4/node, L40S 48 GB ×8/node, H100 MIG slices | main GPU work, benchmarks |
| **Lyra** | EPFL SCITAS | `lyra` → `lyra.hpc.epfl.ch` | B200 ×8/node, RTX PRO 6000 Blackwell 96 GB ×8/node | Blackwell; needs `.venv-blackwell` |
| **Jed** | EPFL SCITAS | `jed.hpc.epfl.ch` | none (CPU) | CPU jobs, DFT, preprocessing |
| **Izar** | EPFL SCITAS | `izar.hpc.epfl.ch` | — | teaching/student accounts only |
| **Daint** | CSCS Alps | `ssh daint` (via ela) | GH200 ×4/node | HPC platform |
| **Clariden** | CSCS Alps | `ssh clariden` | GH200 ×4/node | ML platform (pytorch uenv lives here) |
| **Santis** | CSCS Alps | `ssh santis` | GH200 | climate/weather platform |
| **Eiger** | CSCS Alps | `ssh eiger` | none (AMD CPU) | CPU jobs |
| **sciCORE** | Uni Basel | see [[sciCORE]] | A100, H200, L40S, RTX 4090, Titan | Basel collaborations |

Per-site notes: [[EPFL SCITAS (Kuma, Lyra, Jed)]] · [[CSCS Alps]] · [[sciCORE]]
General: [[SLURM cheatsheet]] · [[Containers on HPC]] · [[Storage and data transfer]] · [[Remote dev on clusters]] · [[Modules and Spack]] · [[SSH and tunnels]]

## Which one?
- Quick GPU test / compile CUDA → Kuma `--qos=debug` (MIG slice if small).
- FP64-heavy (MLIP force training in fp64) → **H100** (Kuma), GH200 (Alps). **Not** L40S / RTX 6000 (weak FP64).
- Multi-node scaling studies → Alps (Slingshot, 4×GH200/node) or Kuma H100 (InfiniBand, NVLink in node).
- Blackwell testing → Lyra (free during beta until 30 Sep 2026; check if billing started).

## Architecture → build flags
| GPU | `sm_` | CPU arch |
|---|---|---|
| L40S | 89 | x86_64 (AMD EPYC 9334) |
| H100 | 90 | x86_64 |
| GH200 | 90 | **aarch64** (Grace) — x86 wheels/binaries won't run |
| B200 | 100 | x86_64 (Intel Xeon 8568Y+) |
| RTX PRO 6000 Blackwell | 120 | x86_64 (AMD EPYC 9535) |

**Alps is ARM.** pip wheels must be `aarch64`; torch for GH200 from `https://download.pytorch.org/whl/cu128` (or the pytorch uenv / NGC container).
