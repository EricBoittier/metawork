# Repo/workspace rules

- **Never push branches/commits to the upstream remotes** of the vendored
  ecosystem repos in this workspace (e.g. `metatensor/metatensor`,
  `metatensor/metatomic`, and similar `upstream` remotes). Only push to the
  user's own forks (`origin`, e.g. `EricBoittier/metatomic`,
  `EricBoittier/symd`).
- **Never reply to GitHub on the user's behalf** — no PR/issue comments,
  no review replies. Reading upstream (fetch, PR/issue lookups via `gh`/API)
  is fine; posting anything is not, unless the user explicitly asks in the
  moment.

# Style

Prefer concise, functional code: names that carry the meaning, lambdas and
comprehensions when they stay readable, and docstrings only when the
behaviour is not obvious from context. Several modules keep short tests at
the bottom of the file and run them on import; match that pattern when it
is already in use.
