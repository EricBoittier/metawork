---
tags: [git, cheatsheet]
---
# Git cheatsheet

See also: [[Fork and upstream workflow]], [[Submodules in metawork]]

## Status / inspect
```bash
git status -sb                      # short status + ahead/behind
git log --oneline --graph -20       # recent history
git log --oneline HEAD..origin/main # what's incoming before a pull
git diff                            # unstaged changes
git diff --staged                   # what will be committed
git show <sha>                      # one commit
git blame -L 10,30 path/file.py     # who changed these lines
```

## Sync
```bash
git fetch --all
git pull --ff-only                  # refuse to make a merge commit
git pull --rebase                   # replay local commits on top
git config pull.rebase true         # make --rebase the default (per repo)
```

## Branches
```bash
git switch -c feat/thing            # new branch
git switch main
git branch -vv                      # branches + tracking
git branch -d old-branch            # delete merged branch (-D to force)
git push -u origin feat/thing       # first push, set upstream
```

## Commit
```bash
git add -p                          # stage hunk by hunk
git commit -m "msg"
git commit --amend --no-edit        # add staged changes to last commit
```

## Rebase
```bash
git rebase -i HEAD~6                # squash/reword/reorder last 6
git rebase -i upstream/main         # rebase onto upstream
git rebase --continue | --skip | --abort
git push --force-with-lease         # after rewriting a pushed branch (safer than -f)
```
Interactive verbs: `p` pick, `r` reword, `s` squash, `f` fixup, `d` drop.

## Stash
```bash
git stash push -m "wip"             # -u to include untracked
git stash list
git stash pop                       # apply + drop
```

## Undo
| Situation | Command |
|---|---|
| Unstage a file | `git restore --staged file` |
| Discard edits to a file | `git restore file` |
| Undo last commit, keep changes | `git reset --soft HEAD~1` |
| Revert a pushed commit | `git revert <sha>` |
| "I lost a commit" | `git reflog` then `git switch -c rescue <sha>` |

## Identity
```bash
git config --global user.name EricBoittier
git config --global user.email eric.boittier@epfl.ch
```
