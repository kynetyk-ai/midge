#!/usr/bin/env python3
"""M1 PR 3 — core correctness and honest errors: #111, #104, #97, #105.

    python3 harness/scenarios/m1_pr3.py

Needs the image built from the branch under test. #111 cannot be provoked on
demand — a model returns an empty turn when it chooses to — so its check plants
one in a real transcript and resumes it against the real API, which is where
the 400 came from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import midgectl

SESSIONS = midgectl.STATE / "rpc-sessions"
results: list[tuple[str, str, str]] = []


def record(name: str, verdict: str, evidence: str) -> None:
    results.append((name, verdict, evidence))
    print(f"{verdict:4}  {name}: {evidence}", flush=True)


def frames_of(frames: list, kind: str) -> list[dict]:
    return [f for f in frames if isinstance(f, dict) and f.get("type") == kind]


def text_of(frames: list) -> str:
    return "".join(f.get("delta", "") for f in frames_of(frames, "assistant_text_delta"))


def session_path() -> str:
    state = midgectl.call("get_state", {})
    assert state and state["success"], state
    return state["data"]["session"]


def check_empty_turn() -> None:
    midgectl.up_quiet()
    midgectl.prompt("Reply with only: hi")
    path = session_path()
    host = SESSIONS / Path(path).name
    empty = {"role": "assistant", "content": [], "model": "gpt-5.4-mini", "stop_reason": "stop"}
    with host.open("a") as f:
        f.write(json.dumps({"type": "message", "data": empty}) + "\n")

    midgectl.up_quiet()
    opened = midgectl.call("open_session", {"path": path})
    assert opened and opened["success"], opened
    frames = midgectl.prompt("Reply with only: still here")
    errors = frames_of(frames, "error")
    ok = not errors and "still here" in text_of(frames).lower()
    record("#111 resume past an empty assistant turn", "PASS" if ok else "FAIL",
           f"errors={[e.get('message', '')[:120] for e in errors]} reply={text_of(frames)!r}")


def check_tool_keyerror() -> None:
    midgectl.up_quiet("--extension-dir", "/opt/midge/examples/notes_extension")
    frames = midgectl.prompt(
        "Use the read_note tool to read the note titled 'Does Not Exist'. "
        "Then tell me exactly what the tool said."
    )
    results_ = frames_of(frames, "tool_result")
    reads = [r["content"] for r in results_ if "not found" in r["content"] or "note" in r["content"].lower()]
    said = [r["content"] for r in results_]
    wrong = any("Tool 'read_note' not found" in c for c in said)
    ok = bool(reads) and not wrong
    record("#104 a tool's KeyError reaches the model as itself", "PASS" if ok else "FAIL",
           f"tool results={said!r}")


def check_edit_near_miss() -> None:
    midgectl.up_quiet()
    frames = midgectl.prompt(
        "Copy /opt/harness/skills/toybox-setting/SKILL.md to notes.md, then use the edit "
        "tool (not bash) on notes.md to change the heading 'The three edits' to 'The "
        "required edits' and reword the sentence under it to match. Change nothing else."
    )
    names = {f["id"]: f["name"] for f in frames_of(frames, "tool_call_end")}
    edits = [r for r in frames_of(frames, "tool_result") if names.get(r["tool_call_id"]) == "edit"]
    misses = [r["content"] for r in edits if r["is_error"]]
    landed = any(not r["is_error"] for r in edits)
    if not edits:
        record("#97 edit near miss names the fix", "UNIT",
               "the model made no edit call; unit-only")
        return
    if not misses:
        record("#97 edit near miss names the fix", "UNIT",
               f"old_text was exact on all {len(edits)} edit call(s); unit-only")
        return
    hinted = all(("whitespace" in m or "closest match" in m or "nothing in the file" in m)
                 for m in misses)
    record("#97 edit near miss names the fix", "PASS" if hinted and landed else "FAIL",
           f"misses={[m[:160] for m in misses]} landed={landed}")


def check_log_dir() -> None:
    try:
        midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
        midgectl._docker(
            "run", "-d", "--name", midgectl.CONTAINER, "--env-file", str(midgectl.REPO / ".env"),
            "-e", "MIDGE_LOG_FILE=/run/midge/nope/deep/midge.log", midgectl.IMAGE,
        )
        midgectl._set_offset(1)
        import time
        time.sleep(2)
        state = midgectl.call("get_state", {})
        logged = midgectl._docker(
            "exec", midgectl.CONTAINER, "sh", "-c", "wc -c < /run/midge/nope/deep/midge.log",
            check=False,
        ).strip()
        ok = bool(state and state["success"]) and logged not in ("", "0")
        record("#105 log file in a missing directory", "PASS" if ok else "FAIL",
               f"get_state={'ok' if state else None} log bytes={logged!r}")
    except Exception as e:
        record("#105 log file in a missing directory", "FAIL", f"{type(e).__name__}: {e}")


def main() -> int:
    for check in (check_empty_turn, check_tool_keyerror, check_edit_near_miss, check_log_dir):
        try:
            check()
        except Exception as e:
            record(check.__name__, "FAIL", f"{type(e).__name__}: {e}")
    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    failed = [r for r in results if r[1] == "FAIL"]
    print(f"\n{len(results) - len(failed)}/{len(results)} without failure")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
