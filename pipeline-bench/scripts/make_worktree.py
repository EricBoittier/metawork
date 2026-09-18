"""Create a detached git worktree for one benchmark variant."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


CONFLICT = re.compile(
    r"<<<<<<<[^\n]*\n(.*?)=======\n(.*?)>>>>>>>[^\n]*\n",
    re.S,
)


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def resolve(repo: Path, ref: str) -> str:
    return git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()


def keep_both_sides(text: str) -> str:
    """Turn add/add conflicts into the concatenation of both sides.

    Independent stacked PRs often each insert a kwarg or changelog bullet at
    the same site; keeping both is the composition we want.
    """
    return CONFLICT.sub(lambda match: match.group(1) + match.group(2), text)


def merge(path: Path, sha: str) -> None:
    result = git(
        path,
        "-c",
        "user.name=pipeline-bench",
        "-c",
        "user.email=pipeline-bench@local",
        "merge",
        "--no-edit",
        sha,
        check=False,
    )
    if result.returncode == 0:
        return
    unmerged = git(path, "diff", "--name-only", "--diff-filter=U").stdout.split()
    if not unmerged:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(result.returncode)
    for rel in unmerged:
        file = path / rel
        resolved = keep_both_sides(file.read_text())
        if "<<<<<<<" in resolved:
            sys.stderr.write(f"could not auto-resolve {rel}\n")
            sys.stderr.write(result.stdout + result.stderr)
            raise SystemExit(result.returncode)
        file.write_text(resolved)
        git(path, "add", rel)
    git(
        path,
        "-c",
        "user.name=pipeline-bench",
        "-c",
        "user.email=pipeline-bench@local",
        "commit",
        "--no-edit",
        "-m",
        f"pipeline-bench: merge {sha[:12]}, keep both sides of add/add conflicts",
    )


def _copy_version(repo: Path, path: Path) -> None:
    """Editable installs generate `_version.py`; worktrees do not have it."""
    src = repo / "src" / "metatrain" / "_version.py"
    dst = path / "src" / "metatrain" / "_version.py"
    if src.is_file() and not dst.is_file():
        dst.write_text(src.read_text())


def materialize(repo: Path, path: Path, refs: list[str]) -> None:
    shas = [resolve(repo, ref) for ref in refs]
    marker = path / ".variant-shas"
    recorded = "\n".join(shas) + "\n"
    if marker.is_file() and marker.read_text() == recorded:
        _copy_version(repo, path)
        return
    if path.exists():
        try:
            git(repo, "worktree", "remove", "--force", str(path))
        except subprocess.CalledProcessError:
            git(repo, "worktree", "prune")
            if path.exists():
                subprocess.run(["rm", "-rf", str(path)], check=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    git(repo, "worktree", "add", "--detach", str(path), shas[0])
    for sha in shas[1:]:
        merge(path, sha)
    _copy_version(repo, path)
    marker.write_text(recorded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--refs", nargs="+", required=True)
    args = parser.parse_args()
    try:
        materialize(args.repo.resolve(), args.path.resolve(), args.refs)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr or str(exc))
        raise SystemExit(exc.returncode)


if __name__ == "__main__":
    main()
