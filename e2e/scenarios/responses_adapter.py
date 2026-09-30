#!/usr/bin/env python3
"""The Responses adapter — `openai` on OpenAI's Responses API, default `gpt-6-luna`.

    python3 e2e/scenarios/responses_adapter.py [check-name ...]

Each check starts its own container, because the provider and model are chosen
at startup. The image pins `MIDGE_MODEL=gpt-5.4-mini`, so the default model is
passed explicitly.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import midgectl

results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, evidence: str) -> None:
    results.append((name, "PASS" if ok else "FAIL", evidence))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {evidence}", flush=True)


def start(env: dict[str, str], *args: str) -> None:
    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    flags = [x for k, v in env.items() for x in ("-e", f"{k}={v}")]
    midgectl._docker(
        "run", "-d", "--name", midgectl.CONTAINER, "--env-file", str(midgectl.REPO / ".env"),
        *flags, midgectl.IMAGE, *args,
    )
    midgectl._set_offset(1)
    time.sleep(2)


def summary(frames: list) -> tuple[list[str], list[str], list[str]]:
    """(tool calls, stop reasons, error messages) in the order they happened."""
    dicts = [f for f in frames if isinstance(f, dict)]
    calls = [f["name"] for f in dicts if f.get("type") == "tool_call_end"]
    stops = [f["stop_reason"] for f in dicts if f.get("type") == "assistant_message_end"]
    errors = [f.get("message", "") for f in dicts if f.get("type") == "error"]
    return calls, stops, errors


def workspace(path: str) -> str:
    return midgectl._docker("exec", midgectl.CONTAINER, "cat", path, check=False)


def check_luna_runs_a_multi_tool_task() -> None:
    start({"MIDGE_MODEL": "gpt-6-luna"})
    frames = midgectl.prompt(
        "Read README.md and src/toybox/text.py, then write FIRST_LINES.md containing the "
        "first line of each file, one per line. Then read FIRST_LINES.md back."
    )
    calls, stops, errors = summary(frames)
    record("gpt-6-luna on `openai`: tools run, no error",
           not errors and "write" in calls and stops[-1:] == ["stop"],
           f"calls={calls} stops={stops} errors={[e[:120] for e in errors]}")
    written = workspace("FIRST_LINES.md")
    record("the file the model wrote exists", bool(written.strip()), repr(written[:80]))

    # A second turn sends the first turn's stored output items back.
    frames = midgectl.prompt("How many lines does FIRST_LINES.md have? Read it to check.")
    calls, stops, errors = summary(frames)
    record("a follow-up turn replays stored output without a 400",
           not errors and stops[-1:] == ["stop"],
           f"calls={calls} stops={stops} errors={[e[:120] for e in errors]}")


def check_chat_completions_is_still_reachable() -> None:
    start({"MIDGE_MODEL": "gpt-5.4-mini", "MIDGE_PROVIDER": "openai-compatible",
           "OPENAI_BASE_URL": "https://api.openai.com/v1"})
    frames = midgectl.prompt("Read README.md and tell me its first heading, nothing else.")
    calls, stops, errors = summary(frames)
    record("`openai-compatible` still speaks chat completions",
           not errors and "read" in calls and stops[-1:] == ["stop"],
           f"calls={calls} stops={stops} errors={[e[:120] for e in errors]}")


def main() -> int:
    checks = (check_luna_runs_a_multi_tool_task, check_chat_completions_is_still_reachable)
    only = sys.argv[1:]
    for check in checks:
        if only and check.__name__ not in only:
            continue
        try:
            check()
        except Exception as e:
            record(check.__name__, False, f"{type(e).__name__}: {e}")
    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    failed = [r for r in results if r[1] != "PASS"]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    print(json.dumps([r[:2] for r in results]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
