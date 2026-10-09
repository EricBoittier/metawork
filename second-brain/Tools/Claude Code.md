---
tags: [tools, claude, ai]
---
# Claude Code

Installed: `~/.local/bin/claude` (2.1.x). `claude update` to upgrade, `claude doctor` if something's off.

## Starting
```bash
claude                          # interactive in current dir
claude "fix the failing test in tests/foo.py"   # start with a prompt
claude -c                       # continue the most recent conversation here
claude -r                       # pick a past conversation to resume (or -r <session-id>)
claude -p "summarise CHANGES.md" # print mode: answer and exit (scriptable)
cat log.txt | claude -p "why did this job fail?"
claude --model opus             # choose model; --effort <level> for reasoning effort
claude -w feat-x                # work in a new git worktree
claude --add-dir ../metatomic   # give access to another directory
claude -n "lammps-debug"        # name the session
```
Background agents: `claude --bg "task"` → `claude agents` (list), `claude attach <id>`, `claude logs <id>`, `claude stop <id>`.

## In the prompt
| Key / prefix | Does |
|---|---|
| `/` | slash commands & skills (type `/` to browse) |
| `!cmd` | run a shell command, output lands in the chat (e.g. `! gh auth login`) |
| `@path` | reference a file/dir |
| `Shift+Tab` | cycle permission modes (normal → auto-accept edits → plan …) |
| `Esc` | interrupt Claude |
| `Esc Esc` | rewind / edit an earlier message |
| `Ctrl+C` twice | quit |
| `\` + Enter | newline |

## Useful slash commands
| Command | Use |
|---|---|
| `/help` | everything available in your version |
| `/clear` | fresh context (new task) |
| `/compact [focus]` | summarise the conversation to free context |
| `/context` | what's using the context window |
| `/model` | switch model |
| `/config` | settings UI |
| `/permissions` | allow/deny tool rules (e.g. `Bash(git status)`) |
| `/memory` | edit CLAUDE.md memory files |
| `/init` | generate a CLAUDE.md for a repo |
| `/resume` | resume a past session |
| `/mcp` | MCP servers status/auth |
| `/agents` | subagents |
| `/hooks` | hook configuration |
| `/code-review`, `/security-review` | review the current diff |
| `/loop 5m <prompt>` | repeat a prompt on an interval |

## Project memory: CLAUDE.md
- `./CLAUDE.md` (checked in) — rules for the repo. metawork's says: never push to upstream remotes, never post on GitHub for me, don't reference fork PR numbers.
- `~/.claude/CLAUDE.md` — personal rules for every project.
- Auto-memory per project in `~/.claude/projects/<path>/memory/`.

## Settings
- `~/.claude/settings.json` (user), `.claude/settings.json` (project, shared), `.claude/settings.local.json` (project, private).
- Permission rules: `"allow": ["Bash(git status:*)", "Bash(uv run pytest:*)"]`.
- `/fewer-permission-prompts` scans history and suggests an allowlist.

## Habits that work
- One task per session; `/clear` between unrelated tasks.
- Say what "done" looks like (tests pass, file X exists).
- Ask for a plan first on big changes (plan mode via Shift+Tab).
- Review diffs before committing; Claude commits/pushes only when asked.
- On clusters: run `claude` inside tmux on the login node, never heavy compute there.

Related: [[tmux]], [[Git cheatsheet]]
