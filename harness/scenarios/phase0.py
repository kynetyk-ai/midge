#!/usr/bin/env python3
"""Phase 0 — the protocol, with no model spend.

Every probe here is answerable without a turn, so nothing reaches the provider.
`compact` is included because an empty history returns at `cut_idx == 0`, before
any provider call.

    python harness/scenarios/phase0.py

Prints one line per probe and writes the full exchange to
`harness/.state/phase0.json` for anything that needs a closer look.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import midgectl  # noqa: E402

Probe = tuple[str, str, dict, str]

# (group, command, params, what a correct midge should do)
PROBES: list[Probe] = [
    # --- the read-only family -------------------------------------------
    ("reads", "get_state", {}, "model, session, message count"),
    ("reads", "get_commands", {}, "10 builtins + any skills"),
    ("reads", "get_messages", {}, "empty history"),
    ("reads", "get_system_prompt", {}, "base + appended halves"),
    ("reads", "get_profiles", {}, "empty; none loaded"),
    ("reads", "get_last_assistant_text", {}, "nothing said yet"),
    ("reads", "list_sessions", {}, "the session this run created"),
    ("reads", "list_sessions", {"roots_only": False}, "same, plus any children"),
    # --- state changes that need no turn ---------------------------------
    ("writes", "set_model", {"model": "gpt-4o-mini"}, "accepted; registry is empty so permissive"),
    ("writes", "set_model", {"model": "gpt-5.4-mini"}, "put it back"),
    ("writes", "set_session_name", {"name": "phase zero"}, "named"),
    ("writes", "set_system_prompt", {"prompt": "You are terse."}, "base replaced, durable"),
    ("writes", "clear_context", {}, "0 messages cleared"),
    ("writes", "compact", {}, "nothing to compact; no provider call"),
    ("writes", "new_session", {"path": "/run/midge/sessions/phase0-new.jsonl"}, "created"),
    ("writes", "open_session", {"path": "/run/midge/sessions/phase0-new.jsonl"}, "no-op, already open"),
    ("writes", "reload", {}, "refused: no sources wired"),
    ("writes", "abort", {}, "refused: nothing in flight"),
    ("writes", "use_profile", {"name": "nope"}, "refused: unknown profile"),
    # --- malformed and hostile -------------------------------------------
    ("bad args", "set_model", {}, "refused: model required"),
    ("bad args", "set_model", {"model": 42}, "refused: must be a string"),
    ("bad args", "set_model", {"model": ""}, "refused: must be non-empty"),
    ("bad args", "new_session", {}, "refused: path required"),
    ("bad args", "set_session_name", {"name": "   "}, "refused: whitespace is not a name"),
    ("bad args", "use_profile", {"name": "x", "transcript": "sideways"}, "refused: not a transcript mode"),
    ("bad args", "use_profile", {"name": "x", "colour": "blue"}, "refused: extra='forbid'"),
    ("bad args", "reload", {"targets": ["everything"]}, "refused: not a reload target"),
    ("bad args", "reload", {"targets": "skills"}, "refused: must be a list"),
    ("bad args", "list_sessions", {"roots_only": "yes"}, "refused: must be a boolean"),
    ("bad args", "steer", {}, "refused: message required"),
    ("bad args", "follow_up", {"message": ""}, "refused: must be non-empty"),
    ("bad args", "prompt", {"message": None}, "refused without starting a run"),
    ("bad args", "nonesuch", {}, "refused: unknown command"),
]

RAW_PROBES: list[tuple[str, str, str]] = [
    ("raw", "{not json", "a parse error, and the loop survives"),
    ("raw", "[1, 2, 3]", "an array is not a command object"),
    ("raw", '"just a string"', "nor is a bare string"),
    ("raw", "null", "nor is null"),
    ("raw", "{}", "an object with no type"),
    ("raw", '{"type": 7}', "a type that is not a string"),
    ("raw", '{"id": {"a": 1}, "type": "get_state"}', "a non-scalar id"),
    ("raw", "   ", "a blank line is not a command"),
]


def _summarise(response: dict | None) -> str:
    if response is None:
        return "NO RESPONSE"
    if response.get("success") is True:
        data = response.get("data")
        rendered = json.dumps(data)
        return f"ok    {rendered[:96]}"
    return f"refused  {response.get('error', '')[:96]}"


def main() -> int:
    results: list[dict] = []
    group = ""
    for grp, command, params, expectation in PROBES:
        if grp != group:
            group = grp
            print(f"\n--- {group} ---")
        response = midgectl.call(command, params, timeout=20.0)
        results.append(
            {"group": grp, "command": command, "params": params,
             "expected": expectation, "response": response}
        )
        arg = json.dumps(params) if params else ""
        print(f"  {command:<24}{arg[:34]:<36}{_summarise(response)}")

    print("\n--- raw ---")
    for grp, line, expectation in RAW_PROBES:
        midgectl.send_raw(line)
        frames = midgectl.read_new(timeout=3.0)
        results.append(
            {"group": grp, "sent": line, "expected": expectation, "frames": frames}
        )
        rendered = json.dumps(frames[0]) if frames else "NO RESPONSE"
        print(f"  {line[:34]:<36}{rendered[:100]}")

    # Alive at the end is the point of the raw probes: none of them should have
    # been able to stop the dispatch loop.
    final = midgectl.call("get_state", {}, timeout=10.0)
    print(f"\nstill serving after every probe: {final is not None and final.get('success')}")

    out = Path(__file__).resolve().parent.parent / ".state" / "phase0.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"full exchange: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
