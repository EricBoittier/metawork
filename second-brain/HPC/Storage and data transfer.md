---
tags: [hpc, storage, data]
---
# Storage and data transfer

## Golden rules
- **Home** = code + configs (small, backed up). **Scratch** = job I/O (fast, **purged**, no backup). **Project/store** = results worth keeping.
- Anything on scratch you'd cry about losing → copy to project storage or back home.
- Many small files kill parallel filesystems (inode quotas, slow metadata) → tar them, or use HDF5/zarr/compressed extxyz shards.
- Write checkpoints to scratch, copy the final/best ones out.

| Site | Home | Scratch (purge) | Long-term |
|---|---|---|---|
| CSCS Alps | `/users/$USER` 50 GB | `$SCRATCH` (30 d; 14 d on clariden) | `$STORE` / `/capstor/store/...` |
| SCITAS | `/home/$USER` | `/scratch/$USER` (purged — check current policy in MOTD/docs) | lab `/work` space (paid) |
| sciCORE | `/scicore/home/<group>/<user>` 1 TB | `$TMPDIR` per job | `/scicore/projects/<p>` |

## rsync (default choice)
```bash
rsync -avhP src/ host:dest/                # trailing / on src = copy contents
rsync -avhP --exclude '*.ckpt' --exclude outputs/ src/ host:dest/
rsync -avhP --dry-run ...                  # preview
rsync -avhP -e "ssh -J ela" big/ daint:'$SCRATCH'/big/   # through jump host
```
`-a` archive, `-v` verbose, `-h` human sizes, `-P` progress + resume partial.

## Other tools
```bash
scp file host:path                         # quick single file
sftp host                                  # interactive
tar czf - dir | ssh host 'tar xzf - -C /dest'   # many small files, one stream
sshfs host:/path ~/mnt/host                # mount remote dir locally (fusermount -u to unmount)
```
Large datasets between centres: Globus if available; otherwise parallel rsync (`find ... | xargs -P8 rsync`).

## Checking usage
```bash
quota                       # CSCS
df -h $HOME                 # sciCORE
du -sh * | sort -h          # what's big here
ncdu ~                      # interactive
find . -type f | wc -l      # inode count
lfs quota -h -u $USER /scratch   # Lustre filesystems
```

## Datasets in metawork
`etc/download-datasets.sh`, `etc/download-madcore-extxyz.sh` — point them at scratch on clusters, not home.
Keep big data **out of git** (see the `.gitignore` lesson in [[Git cheatsheet]]).

Related: [[Clusters overview]], [[SSH and tunnels]]
