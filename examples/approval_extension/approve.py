"""Tool-approval extension: blocks `bash` commands matching a denylist.

Demonstrates the `tool_call` hook — the only lifecycle point that can stop
work before it happens.

    midge --extension-dir examples/approval_extension

An extension participates in the lifecycle by defining `register_hooks`.
The loader calls it with a source-tagged view of the shared `Hooks`, so a
handler that raises names this file in the warning.

The loader also injects `log`, a `midge.ext.approve` logger, so `MIDGE_LOG_LEVEL`
reaches this file too. Never `print()` from an extension: under the TUI, Textual
captures stdout and drops it, and in RPC mode stdout is the protocol.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log: logging.Logger = logging.getLogger("midge.ext.approve")

SYSTEM_PROMPT = (
    "Destructive shell commands are blocked by policy. If a command is "
    "blocked, explain what you were trying to do rather than retrying it."
)

_DENY = re.compile(
    r"\brm\s+-rf?\b|\bgit\s+push\s+--force\b|\bsudo\b|>\s*/dev/[sh]d[a-z]",
    re.IGNORECASE,
)


def register_hooks(hooks: Any) -> None:
    hooks.on("tool_call", _deny_destructive_bash)
    hooks.observe(_audit)


def _deny_destructive_bash(event: Any, ctx: Any) -> Any:
    from midge.hooks import ToolCallResult

    if event.tool_call.name != "bash":
        return None
    command = str(event.tool_call.arguments.get("command", ""))
    match = _DENY.search(command)
    if match is None:
        return None
    log.warning("audit_tool_blocked tool=bash pattern=%s", match.group(0))
    return ToolCallResult(
        block=True,
        reason=f"Blocked by approval policy: {match.group(0)!r} is not permitted.",
    )


def _audit(event: Any, ctx: Any) -> None:
    if event.type == "tool_call":
        log.info("audit_tool_call tool=%s", event.tool_call.name)
