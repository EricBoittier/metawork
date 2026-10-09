---
tags: [tools, editor, vim]
---
# Neovim

`vi`/`vim` are aliased to `~/nvim-linux-x86_64/bin/nvim`. Config: `~/.config-writable/nvim/` (`init.lua`).

## Survival
| Keys | Action |
|---|---|
| `i` / `a` / `o` | insert before / after / new line |
| `Esc` | back to normal mode |
| `:w` `:q` `:wq` `:q!` | save / quit / both / quit without saving |
| `u` / `Ctrl+r` | undo / redo |
| `:e file` | open file |

## Moving
`h j k l` · `w b e` words · `0 ^ $` line start/first char/end · `gg G` top/bottom · `42G` line 42 · `%` matching bracket · `Ctrl+d/u` half page · `*` next occurrence of word · `Ctrl+o/i` jump back/forward.

## Editing (verb + motion)
`dw` delete word · `dd` delete line · `cw` change word · `ci"` change inside quotes · `yy` copy line · `p` paste · `.` repeat · `>>` indent · `J` join lines · `~` toggle case.
Visual: `v` chars, `V` lines, `Ctrl+v` block (then `I` to insert on all lines).

## Search / replace
```vim
/pattern   n N             " search forward, next/prev
:%s/old/new/g              " replace all; add c to confirm each
:%s/\s\+$//e               " strip trailing whitespace
:g/pattern/d               " delete matching lines
```

## Windows, buffers, tabs
`:sp` `:vsp` split · `Ctrl+w h/j/k/l` move · `Ctrl+w =` equalise · `:ls` buffers · `:b name` · `:bd` close buffer · `:tabnew`.

## Macros & registers
`qa … q` record to `a` · `@a` replay · `10@a` ten times · `"+y` copy to system clipboard · `"+p` paste from it.

## Handy
- `:set number relativenumber` · `:set paste` before pasting raw text (old vims)
- `:!cmd` run shell · `:r !date` insert output
- `:term` terminal inside nvim
- `nvim -d a b` diff mode (`]c` next change)
- `:checkhealth` when plugins misbehave
- On clusters without your config: `nvim -u NONE` or plain `vi`.

Related: [[kitty]], [[tmux]]
