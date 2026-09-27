"""`midge --rpc` as a real process: what a client embedding it actually runs.

Every other RPC test drives `RpcServer` in-process. That is how #99 shipped —
the server never wrote a turn to disk, and nothing that ran the server also
read the disk. These spawn the real entrypoint against `tests/fake_openai.py`,
with the real provider, SDK and pipes, and check what crosses the process
boundary: the contract's greeting, a turn, a restart that remembers it, and
malformed input that does not end anything.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from tests.fake_openai import FakeOpenAI

pytestmark = pytest.mark.subprocess

TIMEOUT = 30.0


class Midge:
    """One `python -m midge --rpc` process, spoken to over its pipes."""

    def __init__(self, tmp_path: Path, fake: FakeOpenAI, *args: str) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "MIDGE_"))}
        env |= {
            "OPENAI_BASE_URL": fake.base_url,  # a local server needs no key
            "MIDGE_MODEL": "m",
            "HOME": str(tmp_path),
        }
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "midge", "--rpc", *args],
            cwd=tmp_path,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._frames: queue.Queue[dict[str, Any]] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._frames.put(json.loads(line))

    def send(self, line: dict[str, Any] | bytes) -> None:
        assert self.proc.stdin is not None
        raw = line if isinstance(line, bytes) else json.dumps(line).encode()
        self.proc.stdin.write(raw + b"\n")
        self.proc.stdin.flush()

    def until(self, kind: str) -> list[dict[str, Any]]:
        seen: list[dict[str, Any]] = []
        while True:
            try:
                frame = self._frames.get(timeout=TIMEOUT)
            except queue.Empty:
                err = self.proc.stderr
                stderr = err.read().decode() if err and self.proc.poll() is not None else ""
                raise AssertionError(
                    f"no {kind!r} within {TIMEOUT}s; saw {seen}\n{stderr}"
                ) from None
            seen.append(frame)
            if frame.get("type") == kind:
                return seen

    def close(self) -> int:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        return self.proc.wait(timeout=TIMEOUT)


def test_a_turn_survives_the_process(tmp_path: Path) -> None:
    # The M2 exit criterion: run a turn, end the process, start another on the
    # same session, and find the history intact.
    session = tmp_path / "embedded.jsonl"
    with FakeOpenAI(["Noted: 7341.", "It was 7341."]) as fake:
        first = Midge(tmp_path, fake, "--session", str(session))
        ready = first.until("ready")[-1]
        assert ready["protocol"] == 1
        first.send({"id": "p", "type": "prompt", "message": "remember 7341"})
        turn = first.until("agent_settled")
        assert "".join(f.get("delta", "") for f in turn) == "Noted: 7341."
        assert first.close() == 0

        second = Midge(tmp_path, fake, "--session", str(session))
        second.until("ready")
        second.send({"id": "p", "type": "prompt", "message": "what number?"})
        second.until("agent_settled")
        assert second.close() == 0

    # What the second process sent the model is the proof: the first turn
    # crossed the process boundary through the transcript.
    resumed = [m.get("content") for m in fake.bodies[1]["messages"]]
    assert "remember 7341" in resumed
    assert "Noted: 7341." in resumed
    assert resumed[-1] == "what number?"


def test_malformed_input_never_ends_the_process(tmp_path: Path) -> None:
    with FakeOpenAI([]) as fake:
        midge = Midge(tmp_path, fake, "--no-session")
        midge.until("ready")

        midge.send(b'{"type": "prompt", "message": "' + b"x" * (17 * 1024 * 1024) + b'"}')
        too_long = midge.until("response")[-1]
        midge.send({"id": {"a": 1}, "type": "get_state"})
        bad_id = midge.until("response")[-1]
        midge.send(b"{not json")
        not_json = midge.until("response")[-1]
        midge.send({"id": "s", "type": "get_state"})
        state = midge.until("response")[-1]

        assert "exceeds the 16 MiB limit" in too_long["error"]
        assert bad_id["error"] == "`id` must be a string"
        assert not_json["command"] == "parse"
        assert state["id"] == "s" and state["success"] is True
        assert midge.close() == 0


CLIENT = Path(__file__).resolve().parent.parent / "examples" / "rpc_client.py"


def test_the_example_client_streams_an_answer(tmp_path: Path) -> None:
    with FakeOpenAI(["toybox is a sandbox."]) as fake:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "MIDGE_"))}
        env |= {"OPENAI_BASE_URL": fake.base_url, "MIDGE_MODEL": "m", "HOME": str(tmp_path)}
        server = f"{sys.executable} -m midge --rpc --no-session"
        done = subprocess.run(
            [sys.executable, str(CLIENT), "--server", server, "what is toybox?"],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=TIMEOUT,
        )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "toybox is a sandbox."


def test_the_example_client_refuses_a_protocol_it_does_not_know(tmp_path: Path) -> None:
    stub = tmp_path / "stub.py"
    stub.write_text(
        "import json, sys\n"
        "print(json.dumps({'type': 'ready', 'protocol': 99, 'midge': '9.9'}), flush=True)\n"
        "sys.stdin.read()\n"
    )
    done = subprocess.run(
        [sys.executable, str(CLIENT), "--server", f"{sys.executable} {stub}", "hi"],
        capture_output=True, text=True, timeout=TIMEOUT,
    )
    assert done.returncode == 2
    assert "protocol 99" in done.stderr
