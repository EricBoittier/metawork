---
tags: [obsidian, meta, git]
---
# Syncing this vault

The vault is the `second-brain/` folder of the **metawork** repo (`git@github.com:EricBoittier/metawork.git`). Syncing = git.

## New computer (notes only — recommended)
Sparse checkout: pulls only `second-brain/`, no submodules, no 197 MB of other stuff.
```bash
git clone --no-checkout --filter=blob:none git@github.com:EricBoittier/metawork.git ~/notes-metawork
cd ~/notes-metawork
git sparse-checkout set second-brain
git checkout main
```
Then in Obsidian: vault switcher (bottom-left) → **Open folder as vault** → `~/notes-metawork/second-brain`.

## New computer (full workspace)
```bash
git clone git@github.com:EricBoittier/metawork.git && cd metawork
bash etc/setup-metawork.sh        # repos + venv
```
Open `metawork/second-brain` as a vault.

## Day to day (manual)
```bash
cd <repo> && git pull --rebase                         # before editing
git add second-brain && git commit -m "notes" && git push   # after
```
**Only ever `git add second-brain`** in the full metawork checkout — `git add -A` there picks up checkpoints, nested repos and submodule pointers (that's how the 197 MB "asdf" commit happened).

## Automatic: Obsidian Git plugin
Settings → Community plugins → Turn on → Browse → **Git** (by Vinzent) → Install → Enable. Then in its settings:
- *Advanced → Custom base path*: leave empty if the repo root is found; the vault being a subfolder of the repo is supported.
- *Pull on startup*: on
- *Auto commit-and-sync interval*: 10 (minutes)
- Command palette (Ctrl+P) → "Git: Commit-and-sync" for a manual sync.

⚠️ The plugin stages **everything in the repo**, not just the vault. Use it only on **sparse checkouts** (above), never on the full metawork workspace.

## Phone / tablet
Git on mobile is clunky. Options: Obsidian Sync (paid, easiest), or Syncthing between devices with one desktop doing the git commits.

## Conflicts
Two machines edited the same note → `git pull --rebase` stops:
```bash
git status                      # see conflicted files
# edit file, keep both versions' content, remove <<<<<<< ======= >>>>>>> markers
git add <file> && git rebase --continue && git push
```

## Obsidian troubleshooting (Linux, AppImage)
- **"Command line interface is not enabled"** when launching = another Obsidian is already running (maybe hidden). `pkill -f '.mount_Obsidi'` then relaunch.
- **Graph view blank**, log shows `ZINK ... VK_ERROR_DEVICE_LOST` → GPU/Mesa issue. Launch with `--disable-gpu`, or Settings → Appearance → Advanced → Hardware acceleration off.
- Vault list lives in `~/.config-writable/obsidian/obsidian.json` on cosmopc7 (`XDG_CONFIG_HOME=~/.config-writable`).

Related: [[Home]], [[Git cheatsheet]]
