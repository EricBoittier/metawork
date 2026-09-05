#!/usr/bin/env python3
"""Format/check metatrain Python files after an agent edit (CI ruff)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _file_path(payload: dict) -> Path | None:
    for key in ("file_path", "path", "file", "uri"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            if value.startswith("file://"):
                value = value[7:]
            return Path(value)
    file_obj = payload.get("file")
    if isinstance(file_obj, dict):
        value = file_obj.get("path") or file_obj.get("file_path")
        if isinstance(value, str) and value:
            return Path(value)
    return None


def _ruff_bin(repo_root: Path) -> str | None:
    candidate = repo_root / "metatrain" / ".tox" / "lint" / "bin" / "ruff"
    if candidate.is_file():
        return str(candidate)
    return shutil.which("ruff")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        print("{}")
        return 0

    path = _file_path(payload if isinstance(payload, dict) else {})
    if path is None or path.suffix != ".py":
        print("{}")
        return 0

    resolved = path.resolve()
    text = str(resolved)
    if "/metatrain/" not in text and not text.endswith("/metatrain"):
        print("{}")
        return 0

    repo_root = Path(__file__).resolve().parents[2]
    ruff = _ruff_bin(repo_root)
    if ruff is None:
        print("{}")
        return 0

    env = os.environ.copy()
    subprocess.run([ruff, "format", str(resolved)], check=False, env=env)
    subprocess.run([ruff, "check", "--fix", str(resolved)], check=False, env=env)
    print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
