#!/usr/bin/env python3
"""The example adversarial-reviewer profile has no tool that can change files.

    python3 harness/scenarios/profile_read_only.py [--image midge-test]
"""

from __future__ import annotations

import argparse
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


def tool_calls(frames: list) -> list[str]:
    return [f["name"] for f in frames if isinstance(f, dict) and f.get("type") == "tool_call_end"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=midgectl.IMAGE)
    image = p.parse_args().image
    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    midgectl._docker(
        "run", "-d", "--name", midgectl.CONTAINER, "--env-file", str(midgectl.REPO / ".env"),
        image,
        "--extension-dir", "/opt/midge/examples/profile_extension",
        "--extension-dir", "/opt/midge/examples/approval_extension",
        "--profile", "adversarial-reviewer",
    )
    midgectl._set_offset(1)
    time.sleep(2)
    try:
        resp = midgectl.call("get_profiles", {}) or {}
        prof = next((x for x in resp.get("data", {}).get("profiles", [])
                     if x.get("name") == "adversarial-reviewer"), {})
        tools = prof.get("tools", [])
        record("profile declares only read-only tools", "bash" not in tools and bool(tools),
               f"tools={tools}")
        frames = midgectl.prompt(
            "Use the bash tool to run exactly: echo hi. If you have no bash tool, reply NO_BASH."
        )
        called = tool_calls(frames)
        record("model cannot call bash under the profile", "bash" not in called,
               f"tool calls={called}")
        frames = midgectl.prompt("List the files in the current directory using a tool.")
        called = tool_calls(frames)
        record("read-only tools still work", any(c in ("ls", "grep", "read") for c in called),
               f"tool calls={called}")
    except Exception as e:
        record("scenario", False, f"{type(e).__name__}: {e}")
    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    failed = [r for r in results if r[1] != "PASS"]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    print(json.dumps([r[:2] for r in results]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
