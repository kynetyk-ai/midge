#!/usr/bin/env python3
"""RPC persists turns, resumes them, survives an abort, and compacts.

    python3 e2e/scenarios/rpc_persistence.py

Needs the image built from the branch under test (`midgectl.py up --build`).
Each check recreates the container, so none depends on another's state — except
the resume check, which recreates it *on purpose*, because a restart is the
thing being tested. Transcripts are read from the host mount.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import midgectl

SESSIONS = midgectl.STATE / "rpc-sessions"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, evidence: str) -> None:
    results.append((name, "PASS" if ok else "FAIL", evidence))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {evidence}", flush=True)


def text_of(frames: list) -> str:
    return "".join(
        f.get("delta", "") for f in frames
        if isinstance(f, dict) and f.get("type") == "assistant_text_delta"
    )


def types_of(frames: list) -> list[str]:
    return [f.get("type", "?") for f in frames if isinstance(f, dict)]


def host_path(container_path: str) -> Path:
    return SESSIONS / Path(container_path).name


def messages_on_disk(container_path: str) -> list[dict]:
    lines = host_path(container_path).read_text().splitlines()
    return [r for r in map(json.loads, lines) if r.get("type") == "message"]


def session_of() -> str:
    state = midgectl.call("get_state", {})
    assert state is not None and state["success"], state
    return state["data"]["session"]


def check_resume() -> None:
    midgectl.up_quiet()
    frames = midgectl.prompt("Remember the number 7341. Reply with only: OK")
    path = session_of()
    on_disk = messages_on_disk(path)
    record("turn written to transcript", len(on_disk) >= 2,
           f"{len(on_disk)} message records in {host_path(path).name}")

    midgectl.up_quiet()  # a new process, knowing nothing
    opened = midgectl.call("open_session", {"path": path})
    record("open_session restores the turn",
           bool(opened and opened["success"] and opened["data"]["messages"] >= 2),
           json.dumps(opened and opened.get("data")))

    frames = midgectl.prompt("What number did I ask you to remember? Reply with only the number.")
    answer = text_of(frames).strip()
    record("resumed agent remembers", "7341" in answer, f"answered {answer!r}")

    listed = midgectl.call("list_sessions", {})
    counts = [s["messages"] for s in (listed or {}).get("data", {}).get("sessions", [])
              if s.get("path") == path]
    record("list_sessions counts messages", bool(counts) and counts[0] > 0, f"messages={counts}")


def check_abort() -> None:
    midgectl.up_quiet()
    midgectl.send_raw(json.dumps({
        "id": "p1", "type": "prompt",
        "message": "Use bash to run exactly: sleep 30 && echo finished",
    }))
    midgectl.read_new(timeout=60, until="tool_execution_start")
    time.sleep(1)
    midgectl.call("abort", {})
    frames = midgectl.read_new(timeout=30, until="agent_settled")
    path = session_of()
    on_disk = messages_on_disk(path)
    roles = [m["data"]["role"] for m in on_disk]
    record("aborted turn written to transcript",
           "user" in roles and "tool_result" in roles,
           f"roles on disk: {roles}")

    frames = midgectl.prompt("Say hi in one word.")
    errors = [f for f in frames if isinstance(f, dict) and f.get("type") == "error"]
    record("session usable after abort", not errors and bool(text_of(frames)),
           f"errors={errors!r} reply={text_of(frames)!r}")


def check_compaction_within_a_turn() -> None:
    midgectl.up_quiet("--compaction-threshold", "1000", "--compaction-keep-recent", "500")
    frames = midgectl.prompt(
        "Read each file under src/toybox/ and tests/ one at a time, "
        "then give a one-line summary of each after reading all of them."
    )
    types = types_of(frames)
    end = next((f for f in frames if isinstance(f, dict) and f.get("type") == "compaction_end"), None)
    ordered = ("compaction_end" in types and "agent_end" in types
               and types.index("compaction_end") < types.index("agent_end"))
    record("compaction inside a turn, before agent_end",
           ordered and end is not None and end.get("cut_index") is not None,
           json.dumps(end))
    path = session_of()
    records = [json.loads(line) for line in host_path(path).read_text().splitlines()]
    count = sum(r.get("type") == "compaction" for r in records)
    record("compaction record on disk", count >= 1, f"{count} compaction record(s)")


def check_compaction() -> None:
    midgectl.up_quiet("--compaction-threshold", "3000")
    seen: list[str] = []
    frames: list = []
    for i in range(4):
        frames = midgectl.prompt(
            "Read src/toybox/settings.py, src/toybox/text.py and tests/test_text.py, "
            f"then summarise each in one line. (pass {i})"
        )
        seen += types_of(frames)
        if "compaction_end" in seen:
            break
    end = next((f for f in frames if isinstance(f, dict) and f.get("type") == "compaction_end"),
               None)
    ordered = ("compaction_start" in seen and "compaction_end" in seen
               and seen.index("compaction_start") < seen.index("compaction_end")
               < len(seen) - 1 - seen[::-1].index("agent_settled"))
    record("RPC auto-compacts and says so", ordered and end is not None and "error" not in end,
           json.dumps(end))
    path = session_of()
    records = [json.loads(line) for line in host_path(path).read_text().splitlines()]
    record("compaction record on disk", any(r.get("type") == "compaction" for r in records),
           f"{sum(r.get('type') == 'compaction' for r in records)} compaction record(s)")


def main() -> int:
    for check in (check_resume, check_abort, check_compaction_within_a_turn, check_compaction):
        try:
            check()
        except Exception as e:  # an e2e failure is a result too
            record(check.__name__, False, f"{type(e).__name__}: {e}")
    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    failed = [r for r in results if r[1] != "PASS"]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
