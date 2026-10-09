---
tags: [tools, terminal, tmux]
---
# tmux

Why: sessions survive disconnects — essential on clusters (long `srun --pty`, `claude`, builds). Locally, kitty's tabs/windows usually suffice.

## Sessions (from the shell)
```bash
tmux new -s work            # new named session
tmux ls                     # list
tmux a -t work              # attach (tmux a = last one)
tmux kill-session -t work
tmux new -A -s work         # attach if it exists, else create — good alias
```

## Keys (prefix = `Ctrl+b`, then …)
| Key | Action |
|---|---|
| `d` | detach |
| `s` | session picker · `$` rename session |
| `c` | new window · `,` rename · `&` kill |
| `n` / `p` / `0-9` | next / prev / go to window |
| `w` | tree of windows & sessions |
| `%` / `"` | split vertical / horizontal |
| arrows | move between panes |
| `z` | zoom pane (toggle) |
| `x` | kill pane |
| `Space` | cycle pane layouts |
| `{` / `}` | swap pane |
| `Ctrl+arrow` | resize pane |
| `[` | copy/scroll mode (`/` search, `Space` start, `Enter` copy, `q` quit) |
| `]` | paste |
| `:` | command prompt (e.g. `:setw synchronize-panes on`) |
| `?` | list all keys |

## Suggested `~/.tmux.conf`
```tmux
set -g mouse on
set -g history-limit 50000
set -g base-index 1
setw -g pane-base-index 1
set -g default-terminal "tmux-256color"
set -sg escape-time 10          # snappy Esc in nvim
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
bind r source-file ~/.tmux.conf \; display "reloaded"
```
Reload: `tmux source ~/.tmux.conf`.

## Cluster patterns
```bash
tmux new -A -s gpu
srun -p h100 -q debug --gres=gpu:1 -t 1:00:00 --pty bash   # inside tmux; detach, come back later
```
- Login nodes can be several (kuma1/kuma2): reattach on the same node you started on.
- tmux sessions die when the login node reboots — use sbatch for anything long.
- Logging a pane: `prefix :` → `pipe-pane -o 'cat >> ~/tmux-#S-#I.log'`.

Related: [[Remote dev on clusters]], [[kitty]], [[Claude Code]]
