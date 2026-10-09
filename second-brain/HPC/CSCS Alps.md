---
tags: [hpc, cscs]
---
# CSCS Alps

## Daily login
Keys are short-lived — sign first, then ssh:
```bash
cscs-key sign               # opens browser for MFA
cscs-key sign --headless    # no browser available
ssh clariden                # goes through ela via ProxyJump
```

## Hosts (~/.ssh/config)
| Alias | HostName | Via |
|---|---|---|
| `ela` | ela.cscs.ch | — (jump host) |
| `daint` | daint.alps.cscs.ch | ela |
| `santis` | santis.alps.cscs.ch | ela |
| `eiger` | eiger.alps.cscs.ch | ela |
| `clariden` | clariden.alps.cscs.ch | ela |

Template for a new one:
```
Host NAME
    HostName NAME.alps.cscs.ch
    ProxyJump ela
    User boittier
```

Related: [[SSH and tunnels]], [[Kuma and SLURM]]
