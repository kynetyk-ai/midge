#!/usr/bin/env python3
"""M2 PR 2 — the contract holds against a real model.

    python3 harness/scenarios/m2_pr2.py

`tests/test_rpc_contract.py` pins frame shapes using the fake provider. This
holds the *real* one to the same file: every frame a real turn produces — with
tools, a delivered steer and an abort — must have a shape the golden file
already records. A shape it does not is a contract the tests never saw.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import midgectl

GOLDEN = midgectl.REPO / "tests" / "golden" / "rpc_frames.json"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, evidence: str) -> None:
    results.append((name, "PASS" if ok else "FAIL", evidence))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {evidence}", flush=True)


def shape(frame: dict) -> dict:
    # The same reduction as tests/test_rpc_contract.py, minus the sub-agent
    # envelope, which the golden file does not record per type.
    frame = {k: v for k, v in frame.items() if k != "agent"}
    out: dict = {"type": frame["type"], "keys": sorted(frame)}
    if frame["type"] == "response":
        out["command"] = frame["command"]
        if isinstance(frame.get("data"), dict):
            out["data"] = sorted(frame["data"])
    return out


def main() -> int:
    golden = json.loads(GOLDEN.read_text())
    midgectl.up_quiet()
    frames: list = midgectl.read_new(timeout=10, until="ready")
    first = next((f for f in frames if isinstance(f, dict)), {})
    record("ready is the first frame", first.get("type") == "ready" and first.get("protocol") == 1,
           json.dumps(first))

    state = midgectl.call("get_state", {}) or {}
    data = state.get("data", {})
    record("get_state carries the versions", "protocol" in data and "midge" in data,
           f"protocol={data.get('protocol')} midge={data.get('midge')}")
    frames.append(state)

    # A turn with read and bash, steered while the bash runs.
    midgectl.send_raw(json.dumps({
        "id": "p1", "type": "prompt",
        "message": "Read README.md, then use bash to run exactly: sleep 4 && echo slept. "
                   "Then answer in one sentence.",
    }))
    frames += midgectl.read_new(timeout=90, until='"name": "bash"')
    midgectl.send_raw(json.dumps({"id": "s1", "type": "steer",
                                  "message": "Also say the word 'steered' in your answer."}))
    frames += midgectl.read_new(timeout=120, until="agent_settled")

    # A turn that is aborted.
    midgectl.send_raw(json.dumps({"id": "p2", "type": "prompt",
                                  "message": "Use bash to run exactly: sleep 30"}))
    frames += midgectl.read_new(timeout=90, until="tool_execution_start")
    time.sleep(1)
    midgectl.send_raw(json.dumps({"id": "a2", "type": "abort"}))
    frames += midgectl.read_new(timeout=60, until="agent_settled")

    seen = [f for f in frames if isinstance(f, dict)]
    types = sorted({f["type"] for f in seen})
    unknown = [s for s in {json.dumps(shape(f), sort_keys=True) for f in seen}
               if json.loads(s) not in golden]
    steered = any(f.get("type") == "user_message" and f.get("source") == "steer" for f in seen)
    aborted = any(f.get("type") == "error" and f.get("stop_reason") == "aborted" for f in seen)
    record("real frames all match the golden contract", not unknown,
           f"{len(seen)} frames, types={types}; unknown shapes={unknown}")
    record("the steer was delivered and the abort landed", steered and aborted,
           f"steered={steered} aborted={aborted}")

    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    failed = [r for r in results if r[1] != "PASS"]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
