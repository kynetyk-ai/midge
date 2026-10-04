#!/usr/bin/env python3
"""Smoke-test the built wheel in a clean venv.

    poetry run python scripts/smoke_wheel.py

Builds a wheel, installs it into a throwaway virtual environment, and runs
midge --version and one RPC round trip against it. Standard library only.

Exits non-zero with a one-line reason on the first failure.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 60.0


def _run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, **kw)


def get_version() -> str:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return meta["project"]["version"]


def _smoke(tmp_path: Path) -> int:
    # Step 1: build the wheel
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    out = _run(["poetry", "build", "-f", "wheel", "--output", str(wheel_dir)], cwd=ROOT)
    if out.returncode != 0:
        print(f"build failed: {out.stderr.strip()}", file=sys.stderr)
        return 1
    print("build ok")

    wheels = sorted(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        print(f"expected 1 wheel, found {len(wheels)}", file=sys.stderr)
        return 1

    # Step 2: assert py.typed is inside
    with zipfile.ZipFile(wheels[0]) as zf:
        names = zf.namelist()
    if "midge/py.typed" not in names:
        print("midge/py.typed missing from wheel", file=sys.stderr)
        return 1
    print("py.typed present in wheel")

    # Step 3: create a fresh venv and install the wheel
    venv_dir = tmp_path / "venv"
    out = _run([sys.executable, "-m", "venv", str(venv_dir)])
    if out.returncode != 0:
        print(f"venv failed: {out.stderr.strip()}", file=sys.stderr)
        return 1

    # The venv's pip lives in the bin directory
    bin_dir = venv_dir / "bin"
    pip = bin_dir / "pip"
    out = _run([str(pip), "install", str(wheels[0])])
    if out.returncode != 0:
        print(f"pip install failed: {out.stderr.strip()}", file=sys.stderr)
        return 1
    print("wheel installed into clean venv")

    # Step 4: run midge --version
    midge_bin = bin_dir / "midge"
    ver = get_version()
    out = _run([str(midge_bin), "--version"])
    if out.returncode != 0:
        print(f"--version failed: {out.stderr.strip()}", file=sys.stderr)
        return 1
    expected = f"midge {ver}"
    if out.stdout.strip() != expected:
        print(f"expected {expected!r}, got {out.stdout.strip()!r}", file=sys.stderr)
        return 1
    print(f"--version ok: {expected}")

    # Step 5: assert textual is not installed
    venv_python = bin_dir / "python"
    out = _run([str(venv_python), "-c", 'import importlib.util; exit(1 if importlib.util.find_spec("textual") is not None else 0)'])
    if out.returncode != 0:
        print("textual was installed (should not be)", file=sys.stderr)
        return 1
    print("textual not installed (correct)")

    # Step 6: one RPC round trip
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()

    # Import FakeOpenAI from tests (add tests to sys.path)
    tests_dir = ROOT / "tests"
    sys.path.insert(0, str(tests_dir))
    from fake_openai import FakeOpenAI

    with FakeOpenAI(["pong"]) as fake:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "MIDGE_"))}
        env |= {
            "OPENAI_BASE_URL": fake.base_url,
            "MIDGE_MODEL": "m",
            "HOME": str(home_dir),
        }

        proc = subprocess.Popen(
            [str(midge_bin), "--rpc", "--no-session"],
            cwd=cwd_dir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        frames: queue.Queue[dict] = queue.Queue()

        def _reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if line:
                    with contextlib.suppress(json.JSONDecodeError):
                        frames.put(json.loads(line))

        threading.Thread(target=_reader, daemon=True).start()

        try:
            # Read ready frame
            ready = _wait_frame(frames, "ready", TIMEOUT)
            if not ready:
                print("no ready frame", file=sys.stderr)
                proc.terminate()
                return 1
            print("ready frame received")

            # Send a prompt
            assert proc.stdin is not None
            msg = json.dumps({"id": "1", "type": "prompt", "message": "ping"}).encode()
            proc.stdin.write(msg + b"\n")
            proc.stdin.flush()

            # Read until agent_settled, collecting assistant_text_delta
            assistant_text = []
            while True:
                frame = _wait_frame(frames, None, TIMEOUT)
                if not frame:
                    print("timeout reading frames", file=sys.stderr)
                    proc.terminate()
                    return 1
                if frame.get("type") == "assistant_text_delta":
                    assistant_text.append(frame.get("delta", ""))
                if frame.get("type") == "agent_settled":
                    break

            text = "".join(assistant_text)
            if text != "pong":
                print(f"expected 'pong', got {text!r}", file=sys.stderr)
                proc.terminate()
                return 1
            print("rpc round trip ok: got 'pong'")

        finally:
            proc.stdin.close()
            proc.wait(timeout=TIMEOUT)

    print("all smoke tests passed")
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        return _smoke(Path(tmp))


def _wait_frame(q: queue.Queue, expected_type: str | None, timeout: float) -> dict | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            frame = q.get(timeout=remaining)
            if expected_type is None:
                return frame
            if frame.get("type") == expected_type:
                return frame
        except queue.Empty:
            pass
    return None


if __name__ == "__main__":
    sys.exit(main())
