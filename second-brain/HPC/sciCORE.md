---
tags: [hpc, scicore, basel]
---
# sciCORE (University of Basel)

Docs: https://docs.scicore.unibas.ch · Support: scicore-admin@unibas.ch
> [!todo] Login hostname not on this machine yet — fill in from the docs / your account email, then add a `Host scicore` block to `~/.ssh/config`.

```
Host scicore
    HostName <login node>.scicore.unibas.ch
    User <unibas username>
    IdentityFile ~/.ssh/id_ed25519
```
Usually needs the Uni Basel network / VPN.

## QOS = time limit
Pick the **smallest QOS that fits** — shorter ones schedule faster.
| QOS | Max time |
|---|---|
| `30min` | 30 min |
| `6hours` | 6 h |
| `1day` | 1 day |
| `1week` | 7 days |
| `2weeks` | 14 days |

## GPUs
GPU partitions: `a100`, `h200`, `l40s`, `rtx4090`, `titan`. The QOS must match the partition: `<partition>-<time>`.
```bash
#SBATCH --partition=a100
#SBATCH --qos=a100-6hours
#SBATCH --gres=gpu:1
```
FP64: A100/H200 good; L40S/RTX 4090/Titan weak.

## Batch template
```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --time=06:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x.%j.out      # logs/ must already exist!
#SBATCH --error=logs/%x.%j.err
ml Python/3.11.5-GCCcore-13.2.0      # `ml` = module load
cp input.xyz $TMPDIR/ && cd $TMPDIR  # node-local fast scratch
python run.py
cp results.* $SLURM_SUBMIT_DIR/      # copy back before the job ends — $TMPDIR is wiped
```
Gotchas: `#SBATCH` lines are ignored after the first real command; output dirs aren't created for you.

## Storage
| What | Path | Notes |
|---|---|---|
| Home | `/scicore/home/<group>/<user>` | daily backup; dirs named `*nobackup*` excluded |
| Group share | `/scicore/home/<group>/GROUP` | |
| Projects | `/scicore/projects/<project>` | |
| Node scratch | `$TMPDIR` | fast, wiped after job |
Quota: 1 TB soft / 1.25 TB hard per volume (15-day grace). Check `df -h $HOME`; find hogs with `du -sh *`, `ncdu`, `find $HOME -size +1G`.

## Software
```bash
ml avail / ml spider <name>
ml <software/version>
ml purge
```
Easybuild-style module names (`Toolchain-version`). For our stack prefer a uv venv on top of a `Python` + `CUDA` module.

Related: [[Clusters overview]], [[SLURM cheatsheet]], [[Modules and Spack]]
