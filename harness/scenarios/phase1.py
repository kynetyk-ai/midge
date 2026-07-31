#!/usr/bin/env python3
"""Phase 1 — the four built-in tools, driven by a real model.

    python harness/scenarios/phase1.py [substring-filter]

Each scenario is one prompt. The point is not whether the model succeeds — it is
`gpt-5.4-mini` and often will not — but what midge does when it fails: whether a
bad tool call is answered usefully, whether the loop recovers, and whether the
error text gives the model enough to correct itself.

`reset` recreates the container for scenarios that mutate the workspace, so a
later scenario never inherits an earlier one's damage. Everything else clears the
context between prompts, which is cheaper.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import midgectl  # noqa: E402


@dataclass
class Scenario:
    name: str
    prompt: str
    watching: str
    reset: bool = False
    setup: list[str] = field(default_factory=list)


SCENARIOS: list[Scenario] = [
    # --- read ------------------------------------------------------------
    Scenario("read.ok", "Read src/toybox/text.py and tell me in one sentence what wrap does when width is 0.",
             "a read with sane args, and an answer grounded in the file"),
    Scenario("read.missing", "Read the file src/toybox/nonexistent.py and tell me what it contains.",
             "FileNotFoundError reaches the model as a tool error, and it says so rather than inventing content"),
    Scenario("read.directory", "Read src/toybox and summarise it.",
             "IsADirectoryError, and whether the model recovers by listing instead"),
    Scenario("read.offset", "Show me only lines 30 to 40 of src/toybox/text.py.",
             "whether offset/limit are used or the whole file is read and trimmed by the model"),

    # --- write -----------------------------------------------------------
    Scenario("write.ok", "Create a file called scratch.txt containing exactly the single line: hello",
             "a write with both required args", reset=True),
    Scenario("write.nested", "Create a file at a/b/c/deep.txt containing the word deep.",
             "write mkdir -p behaviour through the model"),

    # --- edit ------------------------------------------------------------
    Scenario("edit.ok", "In src/toybox/text.py change the default width on wrap from 72 to 80. Change nothing else.",
             "an exact old_text, or the near miss #97 predicts", reset=True),
    Scenario("edit.ambiguous", "In src/toybox/text.py replace the word 'word' with 'token' everywhere.",
             "'matches multiple locations' — and whether the model narrows or gives up", reset=True),
    Scenario("edit.missing_file", "Edit src/toybox/nothing.py to change 'x' to 'y'.",
             "FileNotFoundError from edit rather than a silent create"),
    Scenario("edit.stale", "The file src/toybox/tally.py has a function average_length. Change its docstring to say 'Arithmetic mean.' and nothing else.",
             "old_text reproduced from a read, which is where near misses come from", reset=True),

    # --- bash ------------------------------------------------------------
    Scenario("bash.ok", "Run the toybox test suite and tell me how many tests passed.",
             "exit 0, and the model reading the count out of stdout"),
    Scenario("bash.nonzero", "Run: python -m pytest tests/test_tally.py::test_does_not_exist",
             "a non-zero exit appended as [exit code: N], not swallowed"),
    Scenario("bash.timeout", "Run this exactly, with the tool's default timeout: sleep 90",
             "the 60s default firing, SIGTERM then SIGKILL, and a usable message"),
    Scenario("bash.big_output", "Run: python -c \"[print('line %d of noise' % i) for i in range(6000)]\"",
             "the 2000-line / 50KB cap and the spill file the model is told about"),

    # --- the shape #97 was reported against ------------------------------
    Scenario("edit.markdown",
             "Copy /opt/harness/skills/toybox-setting/SKILL.md to notes.md, then in notes.md "
             "change the heading 'The three edits' to 'The required edits' and reword the "
             "sentence under it to match. Change nothing else.",
             "wrapped prose with continuation indents — where #97's near miss came from",
             reset=True),

    # --- combined --------------------------------------------------------
    Scenario("combined.fix", "The README says one function in src/toybox/tally.py is wrong on purpose. Find it, then fix it and run the tests.",
             "read then edit then bash in one turn, and whether the loop holds together", reset=True),
]


def summarise(frames: list) -> dict:
    """What the turn actually did, as distinct from what it said."""
    calls, results, errors = [], [], []
    stops, text = [], []
    for f in frames:
        if not isinstance(f, dict):
            errors.append({"non_json": str(f)[:200]})
            continue
        t = f.get("type")
        if t == "tool_call_end":
            calls.append({"name": f.get("name"), "arguments": f.get("arguments")})
        elif t == "tool_result":
            results.append({
                "is_error": f.get("is_error"),
                "content": (f.get("content") or "")[:240],
            })
        elif t == "assistant_message_end":
            stops.append(f.get("stop_reason"))
        elif t == "error":
            errors.append({"message": f.get("message"), "stop_reason": f.get("stop_reason")})
        elif t == "assistant_text_delta":
            text.append(f.get("delta", ""))
    return {
        "tool_calls": calls,
        "tool_results": results,
        "stop_reasons": stops,
        "errors": errors,
        "final_text": "".join(text).strip()[:400],
    }


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    out: list[dict] = []

    for sc in SCENARIOS:
        if only and only not in sc.name:
            continue
        print(f"\n=== {sc.name} ===")
        print(f"    watching: {sc.watching}")
        if sc.reset:
            midgectl.up_quiet()
        else:
            midgectl.call("clear_context", {}, timeout=15.0)

        frames = midgectl.prompt(sc.prompt, timeout=180.0)
        s = summarise(frames)
        out.append({"scenario": sc.name, "prompt": sc.prompt,
                    "watching": sc.watching, **s})

        for c in s["tool_calls"]:
            args = json.dumps(c["arguments"])
            print(f"    call  {c['name']:<7}{args[:150]}")
        for r in s["tool_results"]:
            flag = "ERR " if r["is_error"] else "ok  "
            print(f"    {flag}  {r['content'][:150]!r}")
        if s["errors"]:
            print(f"    !! {s['errors']}")
        print(f"    stops: {s['stop_reasons']}")
        print(f"    said : {s['final_text'][:160]!r}")

    path = Path(__file__).resolve().parent.parent / ".state" / "phase1.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nfull frames: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
