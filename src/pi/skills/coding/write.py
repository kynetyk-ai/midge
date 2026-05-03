from __future__ import annotations

from pi.skills.coding._helpers import resolve_path
from pi.tools import tool


@tool
async def write(path: str, content: str) -> str:
    """Write content to a file as UTF-8. Always overwrites. Creates parent
    directories as needed.

    Args:
        path: file path (relative paths resolve against the agent's cwd)
        content: file contents
    """
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    p.write_bytes(encoded)
    return f"Wrote {len(encoded)} bytes to {p}"
