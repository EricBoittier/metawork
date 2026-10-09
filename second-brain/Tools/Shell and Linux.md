---
tags: [tools, shell, linux, bash]
---
# Shell and Linux

My aliases (`~/.bashrc`): `cs` = clear + `ls -hrlt` · `ll`/`la`/`l` · `kuma`, `lyra` ssh · `vi`/`vim` → nvim · `kitty`/`kitten`.

## Bash speed
| Keys / syntax | Does |
|---|---|
| `Ctrl+r` | search history (again to go further back) |
| `Ctrl+a` / `Ctrl+e` | line start / end |
| `Ctrl+w` / `Ctrl+u` / `Ctrl+k` | delete word / to start / to end |
| `Alt+.` | insert last argument of previous command |
| `!!` | previous command (`sudo !!`) |
| `!$` | last argument of previous command |
| `cd -` | previous directory |
| `{a,b}` | `cp file{,.bak}` → `cp file file.bak` |
| `$(cmd)` | command substitution |
| `cmd 2>&1 \| tee log` | see and save output |
| `set -euo pipefail` | top of every script: stop on errors |

## Find things
```bash
find . -name '*.py' -newer ref.txt          # by name / time
find . -size +1G -type f                    # big files
grep -rn --include='*.py' 'pattern' src/    # search code (rg is faster if installed)
grep -rl pattern . | xargs sed -i 's/old/new/g'   # replace across files (check first!)
which -a python ; type cs                   # what is this command?
```

## Text crunching
```bash
awk '{print $1, $3}' f          # columns
awk -F, '$3 > 10' data.csv      # filter CSV rows
sort | uniq -c | sort -rn       # frequency count
cut -d: -f1 /etc/passwd
sed -n '10,20p' f               # lines 10–20
head -n 5 f / tail -f log       # follow a log
column -t -s, data.csv          # pretty table
jq '.results[] | {name, energy}' out.json
diff <(sort a) <(sort b)        # compare outputs
```

## Processes
```bash
htop                            # F4 filter, F9 kill, t tree
ps aux | grep '[p]ython'        # brackets avoid matching grep itself
pgrep -af obsidian ; pkill -f pattern ; kill -9 PID
nohup cmd > out.log 2>&1 &      # survive logout (or use tmux)
jobs ; fg %1 ; bg ; disown      # job control (Ctrl+z to suspend)
ss -ltnp                        # who listens on which port
lsof +D dir                     # who has files open in dir
nvidia-smi -l 1 ; watch -n1 nvidia-smi
```

## Disk & files
```bash
df -h ; du -sh * | sort -h ; ncdu
tar czf out.tgz dir/ ; tar xzf out.tgz ; tar tzf out.tgz | head
ln -s target linkname
chmod +x script.sh ; chmod -R g+rX shared/
rsync -avhP src/ dst/
```

## Environment
```bash
export VAR=value ; env | grep CUDA
echo $PATH | tr : '\n'
source ~/.bashrc
ldd $(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib/libtorch.so | grep -i cuda
```

## System (cosmopc, Ubuntu)
```bash
sudo apt update && sudo apt install pkg
nvidia-detector ; nvidia-smi    # driver
journalctl -xe ; dmesg | tail   # what just broke
```

Related: [[tmux]], [[SSH and tunnels]], [[Storage and data transfer]]
