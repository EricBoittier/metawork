---
tags: [tools, github, git]
---
# gh (GitHub CLI)

Reading upstream is fine; **posting on metatensor repos only when I decide to** (see [[Fork and upstream workflow]]).

## Auth
```bash
gh auth login                    # once per machine (SSH protocol)
gh auth status
gh auth refresh -s read:project  # add scopes
```

## Repos
```bash
gh repo clone EricBoittier/metatomic
gh repo fork metatensor/metatomic --clone --remote   # sets origin=fork, upstream=metatensor
gh repo view --web
gh repo sync EricBoittier/metatomic --source metatensor/metatomic   # update fork's default branch
```

## Pull requests
```bash
gh pr list --repo metatensor/metatomic --author @me
gh pr status                     # PRs for current branch / review requests
gh pr checkout 326
gh pr view 326 [--web] [--comments]
gh pr diff 326
gh pr checks [--watch]           # CI status
gh pr create --draft --repo EricBoittier/metatrain --base main --head my-branch --title "..." --body-file body.md
gh pr ready / gh pr merge --squash --delete-branch
```
Use the repo's PR template for the body (`.github/PULL_REQUEST_TEMPLATE/`).

## Issues
```bash
gh issue list --repo metatensor/metatomic --label bug
gh issue view 123
gh search issues "neighbor list" --repo metatensor/metatomic
```

## CI (Actions)
```bash
gh run list --branch my-branch
gh run view <id> --log-failed    # just the failing step logs
gh run watch
gh run rerun <id> --failed
```

## API (anything else)
```bash
gh api repos/metatensor/metatomic/pulls/326/comments | jq '.[].body'
gh api graphql -f query='{ viewer { login } }'
```

Related: [[Git cheatsheet]], [[Fork and upstream workflow]]
