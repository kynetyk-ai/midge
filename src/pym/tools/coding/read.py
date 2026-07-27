from __future__ import annotations

from pym.tools import tool
from pym.tools.coding._helpers import resolve_path

_MAX_LINES = 2000
_MAX_BYTES = 50_000


@tool
async def read(path: str, offset: int = 1, limit: int | None = None) -> str:
    """Read a UTF-8 text file. Returns up to 2000 lines or 50KB, head-truncated.

    Args:
        path: file path (relative paths resolve against the agent's cwd)
        offset: 1-indexed line to start from
        limit: max lines to read (defaults to 2000)
    """
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if p.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path}")

    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)

    start = max(1, offset) - 1
    line_cap = min(limit if limit is not None else _MAX_LINES, _MAX_LINES)
    chunk = lines[start : start + line_cap]

    kept: list[str] = []
    total_bytes = 0
    for line in chunk:
        line_bytes = len(line.encode("utf-8"))
        if not kept and line_bytes > _MAX_BYTES:
            raise ValueError(
                f"First line is {line_bytes} bytes (exceeds {_MAX_BYTES}). "
                "Use sed or head to read this file in smaller chunks."
            )
        if total_bytes + line_bytes > _MAX_BYTES:
            break
        kept.append(line)
        total_bytes += line_bytes

    body = "".join(kept).rstrip("\n")
    next_offset = start + 1 + len(kept)
    truncated = next_offset - 1 < len(lines)
    if truncated:
        body += f"\n\n[truncated; use offset={next_offset} to continue]"
    return body
