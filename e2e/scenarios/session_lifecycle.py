#!/usr/bin/env python3
"""session_start can refuse startup; add_cleanup functions run at exit.

    python3 e2e/scenarios/session_lifecycle.py [--image midge-test]

Uses the fixture extensions in e2e/fixtures/lifecycle{,_gate}/. Needs tmux
for the TUI check.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import midgectl

results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, evidence: str) -> None:
    results.append((name, "PASS" if ok else "FAIL", evidence))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {evidence}", flush=True)


def run_rpc(image: str, sessions: Path, *midge_args: str) -> None:
    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    midgectl._docker(
        "run", "-d", "--name", midgectl.CONTAINER, "--env-file", str(midgectl.REPO / ".env"),
        "-v", f"{sessions}:/run/midge/sessions", image, *midge_args,
    )
    midgectl._set_offset(1)


def check_rpc_gate(image: str) -> None:
    sessions = midgectl.STATE / "lifecycle-gate"
    shutil.rmtree(sessions, ignore_errors=True)
    sessions.mkdir(parents=True)
    run_rpc(image, sessions, "--extension-dir", "/opt/e2e/lifecycle_gate")
    time.sleep(4)
    running = midgectl._running()
    code = midgectl._docker("inspect", "-f", "{{.State.ExitCode}}", midgectl.CONTAINER).strip()
    tmp = sessions / "err"
    midgectl._docker("cp", f"{midgectl.CONTAINER}:{midgectl.RUN}/err", str(tmp), check=False)
    err = tmp.read_text() if tmp.exists() else ""
    record("RPC: a session_start cancel stops startup",
           not running and code != "0" and "cancelled startup" in err,
           f"running={running} exit={code} stderr_tail={err.strip()[-60:]!r}")


def check_rpc_cleanup(image: str) -> None:
    sessions = midgectl.STATE / "lifecycle-rpc"
    shutil.rmtree(sessions, ignore_errors=True)
    sessions.mkdir(parents=True)
    run_rpc(image, sessions, "--extension-dir", "/opt/e2e/lifecycle")
    time.sleep(2)
    state = midgectl.call("get_state", {}) or {}
    midgectl._docker("stop", midgectl.CONTAINER)
    marker = sessions / "cleanup-marker"
    record("RPC: cleanups run when the process exits (SIGTERM)",
           state.get("success") is True and marker.exists(),
           f"get_state ok={state.get('success')} marker={marker.exists()}")


def check_tui_cleanup(image: str) -> None:
    sessions = midgectl.STATE / "tui-sessions"
    marker = sessions / "cleanup-marker"
    marker.unlink(missing_ok=True)
    midgectl.IMAGE = image
    midgectl.main(["tui-up", "--", "--extension-dir", "/opt/e2e/lifecycle"])
    time.sleep(6)
    midgectl.main(["tui-keys", "C-d"])
    deadline = time.monotonic() + 20
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.5)
    midgectl.main(["tui-down"])
    record("TUI: cleanups run when the TUI quits (Ctrl+D)", marker.exists(),
           f"marker={marker.exists()}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=midgectl.IMAGE)
    image = p.parse_args().image
    for check in (check_rpc_gate, check_rpc_cleanup, check_tui_cleanup):
        try:
            check(image)
        except Exception as e:
            record(check.__name__, False, f"{type(e).__name__}: {e}")
    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    failed = [r for r in results if r[1] != "PASS"]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    print(json.dumps([r[:2] for r in results]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
