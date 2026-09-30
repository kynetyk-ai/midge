#!/usr/bin/env python3
"""A copy and an edit of the copy, in one message, run in order (#101).

    python3 e2e/scenarios/tool_call_ordering.py [attempts]

Re-runs the race from builtin_tools.py against a real model. Whether the model puts both calls
in *one* message is its choice, so each attempt reports what it did; only an
attempt that produced a multi-call message with a mutating call in it tests
anything. The order is read from midge.log, where `tool_start` and
`tool_ok`/`tool_failed` carry each call's id.
"""

from __future__ import annotations

import re
import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import midgectl

PROMPT = (
    "Copy /opt/e2e/skills/toybox-setting/SKILL.md to notes.md with bash, and in the "
    "same step use the edit tool on notes.md to change the heading 'The three edits' to "
    "'The required edits'. Issue both tool calls at once."
)
EVENT = re.compile(r"(tool_start|tool_ok|tool_failed) tool=(\S+) id=(\S+)")


def messages(frames: list) -> list[list[tuple[str, str]]]:
    """Tool calls grouped by the assistant message that made them."""
    out: list[list[tuple[str, str]]] = [[]]
    for f in frames:
        if not isinstance(f, dict):
            continue
        if f.get("type") == "tool_call_end":
            out[-1].append((f["id"], f["name"]))
        elif f.get("type") == "assistant_message_end":
            out.append([])
    return [m for m in out if m]


def attempt(n: int) -> str | None:
    midgectl.up_quiet()
    frames = midgectl.prompt(PROMPT)
    log = midgectl._docker("exec", midgectl.CONTAINER, "cat", f"{midgectl.RUN}/midge.log")
    order = [(kind, tid) for kind, _tool, tid in EVENT.findall(log)]
    errors = {f["tool_call_id"]: f["content"] for f in frames
              if isinstance(f, dict) and f.get("type") == "tool_result" and f.get("is_error")}

    together = [m for m in messages(frames) if len(m) > 1]
    if not together:
        shapes = [[name for _, name in m] for m in messages(frames)]
        print(f"attempt {n}: no multi-call message — calls per message: {shapes}")
        return None
    verdicts = []
    for calls in together:
        names = [name for _, name in calls]
        # Each call must have finished before the next one started.
        ordered = _serial(order, [tid for tid, _ in calls])
        failed = [errors[tid][:120] for tid, _ in calls if tid in errors]
        verdicts.append(ordered and not failed)
        print(f"attempt {n}: one message called {names}; serial={ordered} errors={failed}")
    return "PASS" if all(verdicts) else "FAIL"


def _serial(order: list[tuple[str, str]], ids: list[str]) -> bool:
    started: dict[str, int] = {}
    finished: dict[str, int] = {}
    for i, (kind, tid) in enumerate(order):
        (started if kind == "tool_start" else finished).setdefault(tid, i)
    return all(
        finished.get(a, 10**9) < started.get(b, -1) for a, b in pairwise(ids)
    )


def main() -> int:
    attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    verdict = None
    for n in range(1, attempts + 1):
        verdict = attempt(n)
        if verdict is not None:
            break
    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    print(f"\n#101 verdict: {verdict or 'UNIT (model never batched the calls)'}")
    return 1 if verdict == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
