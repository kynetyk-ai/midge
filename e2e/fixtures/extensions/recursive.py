"""Sub-agents the shipped example cannot express, for testing the limits.

`examples/subagent_extension` allows `("read", "bash")`, so it can never reach
the recursion guard — a child only gets a `spawn_*` tool if its own allowlist
names it. `looper` names its own, which is the only way to prove the ancestor
set actually denies the cycle.

`slowpoke` exists to be clamped: it declares a long timeout so a caller asking
for longer still loses to `[subagents] max_timeout`.
"""

from __future__ import annotations

from midge.subagents import subagent


@subagent(
    description="Delegate to another looper. Used only to test recursion refusal.",
    prompt=(
        "You are a test agent. If you have a spawn tool available, call it once "
        "with the same question. If you do not, say exactly: NO SPAWN TOOL."
    ),
    tools=("spawn_looper", "bash"),
    timeout=30,
)
async def looper(question: str) -> str:
    return f"Question: {question}"


@subagent(
    description="Sleeps. Used only to test timeout clamping.",
    prompt="Run exactly the bash command you are given. Report what happened.",
    tools=("bash",),
    timeout=600,
)
async def slowpoke(command: str) -> str:
    return f"Run this and report the outcome: {command}"
