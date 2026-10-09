---
tags: [hpc, remote, tools]
---
# Remote dev on clusters

## Keep sessions alive: tmux
```bash
tmux new -s work          # start
# Ctrl-b d                # detach (job keeps running)
tmux ls
tmux attach -t work
# Ctrl-b c new window · Ctrl-b n/p next/prev · Ctrl-b % split · Ctrl-b [ scroll (q to quit)
```
Login nodes can differ (kuma1/kuma2) — tmux lives on the node you started it on: `ssh kuma1.hpc.epfl.ch` style if needed.
Never run heavy work on the login node; tmux + `srun --pty` into a compute node.

## Jupyter on a compute node
```bash
# on cluster
srun -p h100 -q debug --gres=gpu:1 -c 8 -t 1:00:00 --pty bash
hostname                                   # e.g. kh012
jupyter lab --no-browser --port 8888 --ip=0.0.0.0
# on laptop
ssh -N -L 8888:kh012:8888 kuma             # tunnel through login node
open http://localhost:8888
```
For marimo notebooks: `marimo edit --headless --port 2718` and tunnel 2718.

## VS Code / Cursor Remote-SSH
- Connect to the **login node** for editing only; the server runs there.
- `~/.ssh/config` host entries (`kuma`, `clariden`, ...) show up automatically; ProxyJump works.
- CSCS: run `cscs-key sign` first, or the connection silently fails.
- Put `~/.vscode-server` on a quota-friendly place if home is small (symlink to scratch/project).
- To debug on a GPU node: start `debugpy --listen 5678` in the job, tunnel 5678, attach.

## SSH config conveniences
```
Host *
    ServerAliveInterval 60
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m          # reuse connection → no repeated MFA/password
```

## Web portals
- SCITAS: OnDemand (Jupyter/VS Code in browser), FirecREST API.
- sciCORE: Open OnDemand portal.
- CSCS: JupyterHub on Alps (jupyter-<vcluster>.cscs.ch).

Related: [[SSH and tunnels]], [[Clusters overview]]
