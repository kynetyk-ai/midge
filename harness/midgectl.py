#!/usr/bin/env python3
"""Drive a containerised midge over its JSON-on-stdio RPC.

    python harness/midgectl.py up
    python harness/midgectl.py call get_state
    python harness/midgectl.py prompt "read README.md and say what toybox is"
    python harness/midgectl.py raw '{not json'
    python harness/midgectl.py down

Why it works this way: `serve_stdio` shuts down on stdin EOF, and every
`docker exec` is a separate process — so writing with `docker exec -i` would
close the pipe and kill the server after one command. The container's entrypoint
holds a FIFO open with `sleep infinity`, and this writes into that FIFO.

Output goes to a *file* inside the container rather than a second FIFO, so reads
are `tail -c +OFFSET` from a byte offset kept here. Nothing is lost between
calls and nothing blocks waiting for a writer.

`agent_settled` is the frame to wait on after a prompt: `RpcServer` emits it from
a `finally`, so it arrives on success, on error and on cancellation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

IMAGE = "midge-test"
CONTAINER = "midge-harness"
RUN = "/run/midge"
STATE = Path(__file__).parent / ".state"
REPO = Path(__file__).resolve().parent.parent


class HarnessError(RuntimeError):
    """The container is not in a state the caller can proceed from."""


# --- docker plumbing ------------------------------------------------------


def _docker(*args: str, check: bool = True, stdin: str | None = None) -> str:
    proc = subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        input=stdin,
    )
    if check and proc.returncode != 0:
        raise HarnessError(
            f"docker {' '.join(args[:2])} failed ({proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def _running() -> bool:
    out = _docker(
        "ps", "--filter", f"name=^{CONTAINER}$", "--filter", "status=running",
        "--format", "{{.Names}}", check=False,
    )
    return CONTAINER in out


def _require_running() -> None:
    if not _running():
        raise HarnessError(
            f"container {CONTAINER!r} is not running — `midgectl.py up` first. "
            "If it exited, `midgectl.py logs` shows why."
        )


# --- offset tracking ------------------------------------------------------
#
# One file per container instance. `up` resets it, so a recreated container
# never reads the previous one's frames.


def _offset_file() -> Path:
    STATE.mkdir(exist_ok=True)
    return STATE / "offset"


def _offset() -> int:
    f = _offset_file()
    return int(f.read_text()) if f.exists() else 1


def _set_offset(value: int) -> None:
    _offset_file().write_text(str(value))


# --- the wire -------------------------------------------------------------


def send_raw(line: str) -> None:
    """Write one line to the FIFO, exactly as given.

    `sh -c 'cat >> fifo'` rather than `echo`, so a payload containing quotes,
    backslashes or newlines arrives unmangled — malformed-input probes depend on
    that.
    """
    _require_running()
    _docker("exec", "-i", CONTAINER, "sh", "-c", f"cat >> {RUN}/in", stdin=line + "\n")


def read_new(timeout: float = 0.0, until: str | None = None) -> list[dict | str]:
    """Frames appended since the last read.

    `until` is a substring; polling stops as soon as a line containing it
    arrives. A line that is not JSON is returned as a string rather than being
    dropped — if midge ever writes something unparseable to the protocol
    stream, that is the finding.
    """
    _require_running()
    deadline = time.monotonic() + timeout
    frames: list[dict | str] = []
    start = _offset()
    while True:
        raw = _docker(
            "exec", CONTAINER, "sh", "-c", f"tail -c +{start} {RUN}/out", check=False
        )
        if raw:
            consumed = len(raw.encode("utf-8"))
            for line in raw.splitlines():
                if not line.strip():
                    continue
                try:
                    frames.append(json.loads(line))
                except json.JSONDecodeError:
                    frames.append(line)
            start += consumed
            _set_offset(start)
        if until is not None and any(until in json.dumps(f) for f in frames):
            return frames
        if time.monotonic() >= deadline:
            return frames
        time.sleep(0.25)


def call(command: str, params: dict, timeout: float = 30.0) -> dict | None:
    """Send one command and return the response correlated by `id`."""
    cmd_id = f"h{int(time.monotonic() * 1000) % 100000}"
    send_raw(json.dumps({"id": cmd_id, "type": command, **params}))
    frames = read_new(timeout=timeout, until=f'"{cmd_id}"')
    for f in frames:
        if isinstance(f, dict) and f.get("id") == cmd_id:
            return f
    return None


def prompt(text: str, timeout: float = 180.0) -> list[dict | str]:
    """Send a prompt and read until the turn settles."""
    send_raw(json.dumps({"id": "p1", "type": "prompt", "message": text}))
    return read_new(timeout=timeout, until="agent_settled")


# --- lifecycle ------------------------------------------------------------


def up(args: argparse.Namespace) -> int:
    if args.build:
        print("building…", flush=True)
        proc = subprocess.run(
            ["docker", "build", "-f", "harness/Dockerfile", "-t", IMAGE, "."],
            cwd=REPO,
        )
        if proc.returncode != 0:
            raise HarnessError("image build failed")

    _docker("rm", "-f", CONTAINER, check=False)
    env_file = REPO / ".env"
    if not env_file.exists():
        raise HarnessError(f"{env_file} not found — the container needs OPENAI_API_KEY")

    run_args = [
        "run", "-d", "--name", CONTAINER,
        "--env-file", str(env_file),
        *[x for pair in (("-e", e) for e in args.env) for x in pair],
        IMAGE, *args.midge_args,
    ]
    _docker(*run_args)
    _set_offset(1)

    # A container that dies on startup is the common failure, and it dies fast.
    time.sleep(1.5)
    if not _running():
        print(_docker("logs", CONTAINER, check=False), file=sys.stderr)
        raise HarnessError("container exited immediately — logs above")
    print(f"{CONTAINER} up")
    return 0


def down(_: argparse.Namespace) -> int:
    _docker("rm", "-f", CONTAINER, check=False)
    print(f"{CONTAINER} removed")
    return 0


def logs(_: argparse.Namespace) -> int:
    """Everything the container knows about a failure, in one place."""
    for label, path in (("midge.log", f"{RUN}/midge.log"), ("stderr", f"{RUN}/err")):
        print(f"\n===== {label} =====")
        print(_docker("exec", CONTAINER, "sh", "-c", f"cat {path} 2>/dev/null", check=False))
    print("\n===== docker logs =====")
    print(_docker("logs", CONTAINER, check=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="midgectl")
    sub = p.add_subparsers(dest="cmd", required=True)

    up_p = sub.add_parser("up", help="(re)create the container")
    up_p.add_argument("--build", action="store_true", help="rebuild the image first")
    up_p.add_argument("-e", "--env", action="append", default=[], metavar="K=V")
    up_p.add_argument("midge_args", nargs="*", help="extra args passed to `midge --rpc`")
    up_p.set_defaults(fn=up)

    for name, fn in (("down", down), ("logs", logs)):
        sub.add_parser(name).set_defaults(fn=fn)

    call_p = sub.add_parser("call", help="send one command, print its response")
    call_p.add_argument("command")
    call_p.add_argument("params", nargs="*", metavar="K=V")
    call_p.add_argument("--timeout", type=float, default=30.0)
    call_p.set_defaults(fn=None)

    raw_p = sub.add_parser("raw", help="send a line verbatim")
    raw_p.add_argument("line")
    raw_p.add_argument("--timeout", type=float, default=5.0)
    raw_p.set_defaults(fn=None)

    prompt_p = sub.add_parser("prompt", help="send a prompt, read until settled")
    prompt_p.add_argument("text")
    prompt_p.add_argument("--timeout", type=float, default=180.0)
    prompt_p.set_defaults(fn=None)

    frames_p = sub.add_parser("frames", help="everything since the last read")
    frames_p.add_argument("--timeout", type=float, default=2.0)
    frames_p.set_defaults(fn=None)

    args = p.parse_args(argv)
    try:
        if getattr(args, "fn", None) is not None:
            return args.fn(args)

        if args.cmd == "call":
            params: dict = {}
            for pair in args.params:
                k, _, v = pair.partition("=")
                try:
                    params[k] = json.loads(v)
                except json.JSONDecodeError:
                    params[k] = v
            out = call(args.command, params, timeout=args.timeout)
            print(json.dumps(out, indent=2) if out else "(no response)")
            return 0 if out else 1

        if args.cmd == "raw":
            send_raw(args.line)
            for f in read_new(timeout=args.timeout):
                print(json.dumps(f) if isinstance(f, dict) else f"NON-JSON: {f}")
            return 0

        if args.cmd == "prompt":
            for f in prompt(args.text, timeout=args.timeout):
                print(json.dumps(f) if isinstance(f, dict) else f"NON-JSON: {f}")
            return 0

        if args.cmd == "frames":
            for f in read_new(timeout=args.timeout):
                print(json.dumps(f) if isinstance(f, dict) else f"NON-JSON: {f}")
            return 0
    except HarnessError as e:
        print(f"harness: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
