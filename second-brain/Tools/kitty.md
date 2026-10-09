---
tags: [tools, terminal, kitty]
---
# kitty

Installed at `~/.local/kitty.app` (aliases `kitty`, `kitten` in `~/.bashrc`). Config: `~/.config-writable/kitty/kitty.conf` (this machine sets `XDG_CONFIG_HOME=~/.config-writable`) with `include current-theme.conf`.
Update: `curl -L https://sw.kovidgoyal.net/kitty/installer.sh | sh /dev/stdin`.

## Shortcuts (`kitty_mod` = Ctrl+Shift)
| Keys | Action |
|---|---|
| `Ctrl+Shift+Enter` | new window (split) in current tab |
| `Ctrl+Shift+T` | new tab |
| `Ctrl+Shift+W` / `Q` | close window / tab |
| `Ctrl+Shift+]` / `[` | next / previous window |
| `Ctrl+Shift+→` / `←` | next / previous tab |
| `Ctrl+Shift+.` / `,` | move tab right / left |
| `Ctrl+Shift+Alt+T` | rename tab |
| `Ctrl+Shift+L` | cycle layouts (tall, stack, grid, …) |
| `Ctrl+Shift+R` | resize window mode |
| `Ctrl+Shift+F` / `B` | move window forward / back |
| `Ctrl+Shift+C` / `V` | copy / paste |
| `Ctrl+Shift+H` | scrollback in a pager (searchable with `/`) |
| `Ctrl+Shift+G` | output of the last command in a pager |
| `Ctrl+Shift+Z` / `X` | jump to previous / next prompt |
| `Ctrl+Shift+E` | hints: open a URL on screen |
| `Ctrl+Shift+P` then `F` | hints: pick a file path, insert it |
| `Ctrl+Shift+=` / `-` / `Backspace` | font bigger / smaller / reset |
| `Ctrl+Shift+F2` | edit config |
| `Ctrl+Shift+F5` | reload config |
| `Ctrl+Shift+F11` | fullscreen |
| `Ctrl+Shift+U` | unicode input |
| `Ctrl+Shift+Esc` | kitty shell (run kitty commands) |

## Kittens
```bash
kitten ssh kuma                   # ssh that copies kitty's terminfo + shell integration to the remote
kitten themes                     # pick a theme interactively (writes current-theme.conf)
kitten icat plot.png              # show an image in the terminal (works over kitten ssh)
kitten diff a.py b.py             # side-by-side diff with syntax highlighting
kitten transfer file host:path    # file transfer over the tty (inside kitten ssh)
kitten choose-fonts
```

## Cluster gotcha
On kuma/clariden, plain `ssh` gives `'xterm-kitty': unknown terminal type` (tmux, htop, nvim break).
Fix: use `kitten ssh host` (best — installs terminfo once), or `export TERM=xterm-256color` on the remote. For aliases: `alias kuma='kitten ssh boittier@kuma.hpc.epfl.ch'`.

## Handy config lines
```conf
scrollback_lines 20000
enable_audio_bell no
copy_on_select clipboard
tab_bar_style powerline
enabled_layouts tall,stack,grid
map ctrl+shift+enter new_window_with_cwd
map ctrl+shift+t new_tab_with_cwd
```

Related: [[tmux]], [[SSH and tunnels]]
