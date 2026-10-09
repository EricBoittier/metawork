---
tags: [git, github, protocol]
---
# Fork and upstream workflow

**Rules**
- `origin` = my fork (`EricBoittier/<repo>`), `upstream` = `metatensor/<repo>`.
- Push **only** to `origin`. Never push to upstream.
- Don't write `metatensor/<repo>#<n>` for PRs from my forks in commits/comments — it creates backlink noise.

## Set up remotes
```bash
git remote -v
git remote set-url origin git@github.com:EricBoittier/metatomic.git
git remote add upstream git@github.com:metatensor/metatomic.git
```

## Keep a branch current with upstream
```bash
git fetch upstream
git rebase upstream/main            # or the branch the PR targets
git push --force-with-lease origin HEAD
```

## PRs with gh
```bash
gh auth status
gh pr checkout 326                  # check out someone's PR locally
gh pr list --repo metatensor/metatomic --author @me
gh pr view 326 --web
gh pr create --draft --repo EricBoittier/metatrain --base main --head experimental/lorem \
  --title "..." --body "..."
gh pr checks                        # CI status of current branch's PR
```

Related: [[Git cheatsheet]], [[Conda feedstock protocol]]
