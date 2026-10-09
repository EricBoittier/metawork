---
tags: [metawork, setup]
---
# metawork setup

Repo: `~/Documents/metawork` (`git@github.com:EricBoittier/metawork.git`) — superproject for the metatensor ecosystem.

## (Re)build everything
```bash
bash etc/setup-metawork.sh     # clones/pulls repos, builds .venv, picks torch for the local GPU; safe to re-run
```
Install order: metatensor[torch] → metatomic[torch] → featomic[torch] → metatrain[soap-bpnn,pet] → i-pi → chemiscope …

## Default branches (from setup script / .gitmodules)
| Repo | Branch |
|---|---|
| metatomic | metatomic-core |
| metatrain | experimental/lorem |
| atomistic-cookbook | metatomic-hourglass |
| openmm_meta | master |
| the rest | main |

## Useful scripts in `etc/`
| Script | What |
|---|---|
| `setup-metawork.sh` | full env setup |
| `fix-torch-cuda.sh` | fix torch/CUDA wheel mismatch |
| `rerender-feedstock.sh` | safe conda-smithy rerender → [[Conda feedstock protocol]] |
| `hpc_run.py` | sbatch + manifest → [[Kuma and SLURM]] |
| `download-datasets.sh`, `download-madcore-extxyz.sh` | datasets |

Syncing: [[Submodules in metawork]]
