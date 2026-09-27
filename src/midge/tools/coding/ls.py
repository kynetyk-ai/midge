from __future__ import annotations

from midge.tools import tool
from midge.tools.coding._helpers import resolve_path

_DEFAULT_LIMIT = 500


@tool(read_only=True)
async def ls(path: str = ".", limit: int = _DEFAULT_LIMIT) -> str:
    """List a directory: entries sorted alphabetically, directories marked with a
    trailing '/', dotfiles included. Up to 500 entries unless `limit` says more.

    Args:
        path: directory to list (relative paths resolve against the agent's cwd)
        limit: maximum number of entries to return
    """
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such directory: {path}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    entries = sorted(p.iterdir(), key=lambda e: e.name.lower())
    shown = [e.name + ("/" if e.is_dir() else "") for e in entries[: max(1, limit)]]
    if not shown:
        return "(empty directory)"
    body = "\n".join(shown)
    if len(entries) > len(shown):
        body += f"\n\n[{len(entries) - len(shown)} more; use limit={len(entries)} for all]"
    return body
