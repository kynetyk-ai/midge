"""Allowlist extension: only named tools run, and writes stay under one directory.

    midge --extension-dir examples/allowlist_extension

The companion to `approval_extension`, and the one to copy when a restriction
has to hold. A denylist over `bash` command strings is advisory — the same
effect has a dozen spellings (#102). An allowlist is not: a tool that is not on
it does not run, whatever its arguments say. So the restriction here is the
*set of tools*, and `bash` is deliberately not in it. Add `bash` and everything
below becomes advisory again, because a shell can write wherever the process
can.

The path check on `write` and `edit` is sound only for that reason: those tools
touch exactly the path they are given, and nothing else can write at all.

Edit `ALLOWED` and `WRITABLE` for your own use. Like any hook this applies to
sub-agents too, and it is still no substitute for running midge somewhere that
cannot do damage — a container, or a user without the permission.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log: logging.Logger = logging.getLogger("midge.ext.allowlist")

ALLOWED = frozenset({"read", "ls", "grep", "write", "edit"})
# Relative to the directory midge was started in.
WRITABLE = Path(".")

SYSTEM_PROMPT = (
    "Only these tools are permitted: " + ", ".join(sorted(ALLOWED)) + ". "
    "Files may only be written under the working directory. There is no shell."
)


def register_hooks(hooks: Any) -> None:
    hooks.on("tool_call", _allow)


def _allow(event: Any, ctx: Any) -> Any:
    from midge.hooks import ToolCallResult

    call = event.tool_call
    if call.name not in ALLOWED:
        log.warning("allowlist_blocked tool=%s", call.name)
        return ToolCallResult(block=True, reason=f"{call.name!r} is not an allowed tool.")
    if call.name in ("write", "edit"):
        target = Path(str(call.arguments.get("path", ""))).expanduser()
        root = WRITABLE.resolve()
        if not (Path.cwd() / target).resolve().is_relative_to(root):
            log.warning("allowlist_path_blocked tool=%s path=%s", call.name, target)
            return ToolCallResult(
                block=True, reason=f"{call.name} is only allowed under {root}, not {target}."
            )
    return None
