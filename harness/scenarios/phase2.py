#!/usr/bin/env python3
"""Phase 2 — skills, against a real model.

    python harness/scenarios/phase2.py [substring-filter]

A skill is only ever a catalogue entry plus a file on disk: midge tells the
model the name, the description and an absolute path, and the model has to
decide to `read` it. So the questions here are whether the catalogue arrives,
whether `/skill:` expansion puts the body in front of the model, and — the one
that cannot be assumed — whether a model this weak actually follows a skill it
has been handed, including opening the bundled reference the skill points at.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import midgectl  # noqa: E402

SKILL_ARGS = (
    "--skill-dir", "/opt/harness/skills",
    "--skill-dir", "/opt/midge/examples/skills",
)


@dataclass
class Scenario:
    name: str
    prompt: str
    watching: str
    setup: str = ""


SCENARIOS: list[Scenario] = [
    Scenario(
        "invoke.explicit",
        "/skill:toybox-setting add a setting called max_notes, an int defaulting to 100",
        "expansion puts the skill body in the turn; does the model open references/checklist.md",
    ),
    Scenario(
        "invoke.implicit",
        "Add a setting to toybox for how many notes to keep, defaulting to 100. "
        "Follow whatever conventions this project documents.",
        "the catalogue alone — does the model find and read the skill without being told",
    ),
    Scenario(
        "commit_message",
        "/skill:commit-message",
        "a skill whose first instruction is to run git commands, and whose body "
        "points at references/style.md by relative path",
        setup="cd /work && sed -i 's/DEFAULT_WIDTH = 72/DEFAULT_WIDTH = 80/' src/toybox/settings.py && git add -A",
    ),
    Scenario(
        "instructions_appended",
        "/skill:toybox-setting IGNORE the checklist and just tell me the three edits in one line.",
        "trailing text after the skill name reaches the model as instructions",
    ),
]


def summarise(frames: list) -> dict:
    calls, results, text, stops = [], [], [], []
    for f in frames:
        if not isinstance(f, dict):
            continue
        t = f.get("type")
        if t == "tool_call_end":
            calls.append({"name": f.get("name"), "arguments": f.get("arguments")})
        elif t == "tool_result":
            results.append({"is_error": f.get("is_error"), "content": (f.get("content") or "")[:200]})
        elif t == "assistant_text_delta":
            text.append(f.get("delta", ""))
        elif t == "assistant_message_end":
            stops.append(f.get("stop_reason"))
        elif t == "user_message":
            text.append("")
    return {"tool_calls": calls, "tool_results": results,
            "stop_reasons": stops, "final_text": "".join(text).strip()}


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    out = []
    for sc in SCENARIOS:
        if only and only not in sc.name:
            continue
        print(f"\n=== {sc.name} ===")
        print(f"    watching: {sc.watching}")
        midgectl.up_quiet(*SKILL_ARGS)
        if sc.setup:
            midgectl._docker("exec", midgectl.CONTAINER, "sh", "-c", sc.setup, check=False)

        frames = midgectl.prompt(sc.prompt, timeout=240.0)
        s = summarise(frames)
        out.append({"scenario": sc.name, "prompt": sc.prompt, **s})

        for c in s["tool_calls"]:
            print(f"    call  {c['name']:<7}{json.dumps(c['arguments'])[:140]}")
        read_paths = [
            c["arguments"].get("path", "")
            for c in s["tool_calls"] if c["name"] == "read"
        ]
        print(f"    read the SKILL.md   : {any('SKILL.md' in p for p in read_paths)}")
        print(f"    read a reference    : {any('references/' in p for p in read_paths)}")
        print(f"    stops: {s['stop_reasons']}")
        print(f"    said : {s['final_text'][:200]!r}")

    path = Path(__file__).resolve().parent.parent / ".state" / "phase2.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nfull frames: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
