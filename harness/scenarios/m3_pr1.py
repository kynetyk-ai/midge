#!/usr/bin/env python3
"""M3 PR 1 — the core stops assuming it is a coding agent.

    python3 harness/scenarios/m3_pr1.py

Each check starts its own container, because what is being tested is how midge
is configured at startup: `[agent] system_prompt`, `[tools] builtin`, and the
skills catalogue's reader.
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


def system_prompt() -> str:
    resp = midgectl.call("get_system_prompt", {}) or {}
    return resp.get("data", {}).get("prompt", "")


def tool_names(frames: list) -> list[str]:
    return [f["name"] for f in frames if isinstance(f, dict) and f.get("type") == "tool_call_end"]


def check_a_different_identity_with_no_coding_tools() -> None:
    start({"MIDGE_BUILTIN_TOOLS": "false",
           "MIDGE_SYSTEM_PROMPT": "You are Quill, a librarian. You have no tools."})
    prompt = system_prompt()
    log = midgectl._docker("exec", midgectl.CONTAINER, "cat", f"{midgectl.RUN}/midge.log")
    record("configured identity is the base prompt", prompt.startswith("You are Quill"),
           prompt[:60])
    record("no built-in tools loaded", "tools=0" in log,
           next((line for line in log.splitlines() if "startup mode=" in line), "?")[-80:])
    frames = midgectl.prompt("Use bash to run exactly: echo hi. If you cannot, say why in one sentence.")
    called = tool_names(frames)
    reply = "".join(f.get("delta", "") for f in frames
                    if isinstance(f, dict) and f.get("type") == "assistant_text_delta")
    record("no bash to call", not called, f"tool calls={called} reply={reply[:120]!r}")


def check_the_catalogue_names_the_reader() -> None:
    start({"MIDGE_BUILTIN_TOOLS": "read"}, "--skill-dir", "/opt/harness/skills")
    prompt = system_prompt()
    record("catalogue shown and names `read`",
           "<available_skills>" in prompt and "Use the read tool" in prompt,
           f"available_skills={'<available_skills>' in prompt}")
    frames = midgectl.prompt(
        "I want to add a new toybox setting. Before doing anything, open the skill that "
        "describes how, and tell me its first step in one sentence."
    )
    reads = [f["arguments"].get("path", "") for f in frames if isinstance(f, dict)
             and f.get("type") == "tool_call_end" and f.get("name") == "read"]
    record("model opens a SKILL.md with the declared reader",
           any(p.endswith("SKILL.md") for p in reads), f"read paths={reads}")


def check_a_profile_without_a_reader_gets_no_catalogue() -> None:
    start({}, "--extension-dir", "/opt/midge/examples/approval_extension",
          "--extension-dir", "/opt/harness/extensions",
          "--skill-dir", "/opt/harness/skills", "--profile", "blind")
    prompt = system_prompt()
    record("profile `blind` (bash only) gets no catalogue at startup",
           "available_skills" not in prompt, f"available_skills in prompt={'available_skills' in prompt}")


def check_spill_prefix() -> None:
    start({})
    frames = midgectl.prompt("Use bash to run exactly: seq 1 20000. Reply with only: done")
    results_ = [f.get("content", "") for f in frames
                if isinstance(f, dict) and f.get("type") == "tool_result"]
    spill = next((r for r in results_ if "spilled to" in r), "")
    record("#108 spill file is midge_bash_*", "/midge_bash_" in spill, spill[:90])


def main() -> int:
    checks = (check_a_different_identity_with_no_coding_tools, check_the_catalogue_names_the_reader,
              check_a_profile_without_a_reader_gets_no_catalogue, check_spill_prefix)
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
