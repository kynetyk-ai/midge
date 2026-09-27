#!/usr/bin/env python3
"""M2 PR 1 — no malformed input ends the process: #100, #106, #107, #109.

    python3 harness/scenarios/m2_pr1.py

No model spend: every probe is answerable without a turn.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import midgectl

results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, evidence: str) -> None:
    results.append((name, "PASS" if ok else "FAIL", evidence))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {evidence}", flush=True)


def responses(frames: list) -> list[dict]:
    return [f for f in frames if isinstance(f, dict) and f.get("type") == "response"]


def check_over_long_line() -> None:
    midgectl.up_quiet()
    midgectl.send_raw('{"id":"big","type":"prompt","message":"' + "x" * (17 * 1024 * 1024) + '"}')
    refused = responses(midgectl.read_new(timeout=20, until="exceeds"))
    alive = midgectl._running()
    state = midgectl.call("get_state", {}) if alive else None
    record("#100 a 17 MiB line is refused, not fatal",
           bool(refused) and alive and bool(state and state["success"]),
           f"refusal={[r.get('error') for r in refused]} running={alive} "
           f"get_state={'ok' if state and state.get('success') else None}")


def check_non_string_id() -> None:
    midgectl.send_raw(json.dumps({"id": {"a": 1}, "type": "get_state"}))
    [resp, *_] = responses(midgectl.read_new(timeout=5, until="response")) or [{}]
    record("#106 a non-string id is refused",
           resp.get("success") is False and resp.get("error") == "`id` must be a string",
           json.dumps(resp))


def check_readable_refusals() -> None:
    a = midgectl.call("use_profile", {"name": "x", "transcript": "sideways"}) or {}
    b = midgectl.call("reload", {"targets": "skills"}) or {}
    c = midgectl.call("prompt", {"message": "/skill:nope"}) or {}
    errors = [r.get("error", "") for r in (a, b, c)]
    ok = all("\n" not in e and "Params" not in e for e in errors) and not errors[2].startswith('"')
    record("#107 refusals are one readable line", ok, json.dumps(errors))


def check_bound_once() -> None:
    midgectl.up_quiet("--extension-dir", "/opt/midge/examples/subagent_extension")
    midgectl.call("get_state", {})
    log = midgectl._docker("exec", midgectl.CONTAINER, "cat", f"{midgectl.RUN}/midge.log")
    n = log.count("subagents_bound")
    record("#109 sub-agents bound once", n == 1, f"subagents_bound lines={n}")


def main() -> int:
    for check in (check_over_long_line, check_non_string_id, check_readable_refusals,
                  check_bound_once):
        try:
            check()
        except Exception as e:
            record(check.__name__, False, f"{type(e).__name__}: {e}")
    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    failed = [r for r in results if r[1] != "PASS"]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
