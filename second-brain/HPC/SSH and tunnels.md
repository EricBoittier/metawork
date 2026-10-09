---
tags: [ssh, network]
---
# SSH and tunnels

## Keys
```bash
ssh-keygen -t ed25519 -C "eric.boittier@epfl.ch" -f ~/.ssh/me.cosmolab
ssh-copy-id -i ~/.ssh/me.cosmolab boittier@cosmopc27.epfl.ch
# all lab PCs at once:
for pc in cosmopc3 cosmopc16 cosmopc24 cosmopc27; do ssh-copy-id -i ~/.ssh/me.cosmolab boittier@$pc.epfl.ch; done
```
GitHub: `ssh -T git@github.com` to test.

## Port forwarding (Jupyter etc.)
```bash
# on remote: jupyter lab --no-browser --port 8888
ssh -CNL localhost:8888:localhost:8888 boittier@cosmopc27    # then open http://localhost:8888
ssh -CNL localhost:5678:localhost:5678 boittier@cosmopc27    # debugpy
```
`-C` compress, `-N` no shell, `-L local:host:remote`.
Through a compute node: `ssh -J kuma -CNL 8888:localhost:8888 <node>`.

## Debug a connection
```bash
ssh -vvv host
```

## Copy files
```bash
rsync -avzP src/ host:dest/          # resumable, trailing slash = contents
scp file host:path
```

Related: [[CSCS Alps]], [[Kuma and SLURM]]
