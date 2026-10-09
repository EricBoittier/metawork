---
tags: [git, submodules, metawork, protocol]
---
# Submodules in metawork

`~/Documents/metawork` is a superproject; each ecosystem repo is a submodule that I often have on my own feature branch.
`git submodule update` would **detach and move** those checkouts, so don't run it blindly.

## Read the status column
```bash
git submodule status
```
- ` ` (space) — checked out at the recorded commit
- `+` — checked out at a *different* commit than recorded (usually my branch)
- `-` — not initialised
- `U` — merge conflict

## Safe sync protocol
1. Pull the superproject:
   ```bash
   git fetch --all && git log --oneline HEAD..origin/main
   git pull --ff-only
   ```
2. Classify each `+` submodule (clean? behind/ahead/diverged from the recorded commit):
   ```bash
   for s in $(git submodule status | awk '/^\+/{print $2}'); do
     rec=$(git ls-tree HEAD $s | awk '{print $3}'); cur=$(git -C $s rev-parse HEAD)
     dirty=$(git -C $s status --porcelain | wc -l)
     if git -C $s merge-base --is-ancestor $cur $rec; then rel=behind
     elif git -C $s merge-base --is-ancestor $rec $cur; then rel=ahead
     else rel=diverged; fi
     echo "$s branch=$(git -C $s branch --show-current) dirty=$dirty $rel"
   done
   ```
3. Act:
   - **behind + clean** → fast-forward: `git -C <s> merge --ff-only <rec>` (on a branch) or `git -C <s> checkout <rec>` (detached)
   - **ahead** → my newer work; commit the pointer if I want it recorded: `git add <s> && git commit`
   - **diverged / dirty** → leave it; deal with it by hand.

## Other useful bits
```bash
git submodule update --init <path>        # init a single missing one
git submodule foreach 'git status -sb'    # quick look at all
git -C <s> fetch && git -C <s> log --oneline HEAD..@{u}
```

Feedstock clones live **next to** metawork, never inside it (a nested clone gets recorded as a gitlink) — see [[Conda feedstock protocol]].
