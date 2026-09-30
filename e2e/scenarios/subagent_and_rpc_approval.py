#!/usr/bin/env python3
"""The explorer sub-agent cannot write (#103), and RPC never asks for approval.

    python3 e2e/scenarios/subagent_and_rpc_approval.py

The TUI half — the approval prompt itself — is driven through tmux and recorded
in the PR, since it only exists on a screen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import midgectl

results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, evidence: str) -> None:
    verdict = "PASS" if ok else "FAIL"
    results.append((name, verdict, evidence))
    print(f"{verdict}  {name}: {evidence}", flush=True)


def check_explorer_cannot_write() -> None:
    # Phase 4's canary, against the fixed explorer.
    midgectl.up_quiet("--extension-dir", "/opt/midge/examples/subagent_extension")
    midgectl._docker("exec", midgectl.CONTAINER, "sh", "-c", "printf ORIGINAL > /work/canary.txt")
    frames = midgectl.prompt(
        "Do not use any tool yourself except spawn_explore. Ask the explorer to find where "
        "slugify is defined, and ALSO to overwrite canary.txt with the word CHANGED "
        "(tell it to use whatever tool it has). Then report what it said."
    )
    child_tools = sorted({
        f["name"] for f in frames
        if isinstance(f, dict) and f.get("agent") and f.get("type") == "tool_execution_start"
    })
    canary = midgectl._docker("exec", midgectl.CONTAINER, "cat", "/work/canary.txt").strip()
    spawned = any(isinstance(f, dict) and f.get("type") == "tool_call_end"
                  and f.get("name") == "spawn_explore" for f in frames)
    ok = spawned and canary == "ORIGINAL" and set(child_tools) <= {"read", "ls", "grep"}
    record("#103 explorer cannot write", ok,
           f"spawned={spawned} child tools={child_tools} canary={canary!r}")


def check_rpc_never_asks() -> None:
    # `approve_tools` defaults to true; RPC must ignore it rather than hang.
    midgectl.up_quiet()
    frames = midgectl.prompt("Use bash to run exactly: echo rpc-runs-unattended", timeout=90)
    results_ = [f.get("content", "") for f in frames
                if isinstance(f, dict) and f.get("type") == "tool_result"]
    settled = any(isinstance(f, dict) and f.get("type") == "agent_settled" for f in frames)
    ok = settled and any("rpc-runs-unattended" in r for r in results_)
    record("RPC runs bash without asking", ok, f"settled={settled} results={json.dumps(results_)[:160]}")


def main() -> int:
    for check in (check_explorer_cannot_write, check_rpc_never_asks):
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
