from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import subprocess
from pathlib import Path

from midge.tools import tool
from midge.tools.coding._helpers import resolve_path

_DEFAULT_LIMIT = 100
_MAX_BYTES = 50_000
_MAX_LINE = 500


@tool(read_only=True)
async def grep(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    ignore_case: bool = False,
    literal: bool = False,
    limit: int = _DEFAULT_LIMIT,
) -> str:
    """Search file contents. Returns `path:line: text` for each matching line.

    Inside a git repository only tracked and unignored files are searched, so
    `.gitignore` is respected. Binary files are skipped. Output stops at 100
    matches (or `limit`) or 50KB; lines longer than 500 characters are cut.

    Args:
        pattern: a Python regular expression, or plain text with `literal=true`
        path: directory or file to search (default: the agent's cwd)
        glob: only files whose path matches, e.g. '*.py' or 'tests/*'
        ignore_case: match regardless of case
        literal: treat `pattern` as plain text rather than a regex
        limit: maximum number of matching lines to return
    """
    root = resolve_path(path)
    if not root.exists():
        raise FileNotFoundError(f"No such file or directory: {path}")
    try:
        rx = re.compile(re.escape(pattern) if literal else pattern, re.I if ignore_case else 0)
    except re.error as e:
        raise ValueError(f"Invalid regex {pattern!r}: {e}. Pass literal=true to search text.") from e

    # A search walks the disk; off the event loop so a big tree does not stall
    # every other task — the TUI's rendering among them.
    return await asyncio.to_thread(_search, root, rx, glob, max(1, limit))


def _search(root: Path, rx: re.Pattern[str], glob: str | None, limit: int) -> str:
    base = root if root.is_dir() else root.parent
    out: list[str] = []
    size = 0
    for f in [root] if root.is_file() else _files(root):
        rel = os.path.relpath(f, base)
        if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(f.name, glob)):
            continue
        try:
            data = f.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue
        for n, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            if not rx.search(line):
                continue
            text = line if len(line) <= _MAX_LINE else line[:_MAX_LINE] + "…"
            entry = f"{rel}:{n}: {text}"
            size += len(entry) + 1
            if len(out) >= limit or size > _MAX_BYTES:
                return "\n".join(out) + f"\n\n[stopped at {len(out)} matches; narrow the search]"
            out.append(entry)
    return "\n".join(out) if out else "No matches."


def _files(root: Path) -> list[Path]:
    # git knows what is ignored; reimplementing .gitignore semantics here would
    # be the larger and wronger half of this tool.
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
        return sorted(root / name for name in listed.decode().split("\0") if name)
    except (OSError, subprocess.CalledProcessError):
        pass
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != ".git")
        found.extend(Path(dirpath) / name for name in sorted(filenames))
    return found
