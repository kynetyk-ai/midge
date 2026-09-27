#!/usr/bin/env python3
"""Drive a containerised midge over its JSON-on-stdio RPC.

    python3 harness/midgectl.py up
    python3 harness/midgectl.py call get_state
    python3 harness/midgectl.py prompt "read README.md and say what toybox is"
    python3 harness/midgectl.py raw '{not json'
    python3 harness/midgectl.py down

    python3 harness/midgectl.py tui-up -- --extension-dir /opt/midge/examples/approval_extension
    python3 harness/midgectl.py tui-type "read README.md" && python3 harness/midgectl.py tui-keys Enter
    python3 harness/midgectl.py tui-screen --until "toybox" --timeout 60
    python3 harness/midgectl.py tui-down

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
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

IMAGE = "midge-test"
CONTAINER = "midge-harness"
TUI_CONTAINER = "midge-tui"
TMUX_SESSION = "midge-tui"
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
        *_sessions_mount("rpc-sessions"),
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


def up_quiet(*midge_args: str) -> None:
    """Recreate the container, optionally with different `midge --rpc` args.

    Scenarios that mutate the workspace use this to get a clean one; scenarios
    that need different extensions or skills loaded use it to change what the
    process was started with, since neither can be altered after startup.
    """
    _docker("rm", "-f", CONTAINER, check=False)
    _docker(
        "run", "-d", "--name", CONTAINER,
        "--env-file", str(REPO / ".env"), *_sessions_mount("rpc-sessions"),
        IMAGE, *midge_args,
    )
    _set_offset(1)
    time.sleep(1.5)
    if not _running():
        raise HarnessError("container exited on restart")


def _sessions_mount(name: str) -> list[str]:
    """Transcripts land on the host, so they outlive the container.

    `up` recreates the container, which would otherwise discard every
    transcript — and "restart, then resume" is exactly what needs testing.
    """
    host = STATE / name
    host.mkdir(parents=True, exist_ok=True)
    return ["-v", f"{host.resolve()}:/run/midge/sessions"]


def _tui_command(args: argparse.Namespace) -> list[str]:
    # argparse leaves the `--` separator in REMAINDER; midge would reject it.
    extra = args.midge_args[1:] if args.midge_args[:1] == ["--"] else args.midge_args
    return [
        "docker", "run", "-it", "--rm",
        # Docker's default detach sequence is Ctrl+P, Ctrl+Q, so it swallows
        # Ctrl+P — the TUI's command list — before midge ever sees it.
        "--detach-keys", "ctrl-^",
        "--name", TUI_CONTAINER,
        "--env-file", str(REPO / ".env"),
        "-e", f"MIDGE_HARNESS_CONFIG={args.config}",
        *_sessions_mount("tui-sessions"),
        *[x for pair in (("-e", e) for e in args.env) for x in pair],
        "--entrypoint", "/usr/local/bin/tui-entrypoint.sh",
        IMAGE, *extra,
    ]


def tui(args: argparse.Namespace) -> int:
    """Print the `docker run` line for an interactive TUI session.

    For a person at a terminal. `tui-up` runs the same line inside tmux, which
    is how a script — or an agent — drives the TUI without owning a TTY.
    """
    print(shlex.join(_tui_command(args)))
    print(f"\n# transcripts will appear in {STATE / 'tui-sessions'}", file=sys.stderr)
    return 0


# --- the TUI, through tmux -------------------------------------------------
#
# Textual needs a terminal; tmux is one that can be written to and read from by
# another process. `capture-pane` returns the rendered screen as text, which is
# what a check asserts on — not the byte stream Textual emitted to draw it.


def _tmux(*args: str, check: bool = True) -> str:
    proc = subprocess.run(["tmux", *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise HarnessError(f"tmux {args[0]} failed: {proc.stderr.strip()}")
    return proc.stdout


def tui_up(args: argparse.Namespace) -> int:
    _tui_down()
    _tmux(
        "new-session", "-d", "-s", TMUX_SESSION, "-x", str(args.width), "-y", str(args.height),
        shlex.join(_tui_command(args)),
    )
    # Keep a pane whose process died, so the traceback is still readable.
    _tmux("set-option", "-t", TMUX_SESSION, "remain-on-exit", "on")
    print(f"{TMUX_SESSION} up — `tui-screen` to look")
    return 0


def tui_keys(args: argparse.Namespace) -> int:
    """tmux key names: Enter, Escape, C-c, M-Enter, C-o, Up, …"""
    _tmux("send-keys", "-t", TMUX_SESSION, *args.keys)
    return 0


def tui_type(args: argparse.Namespace) -> int:
    # As a bracketed paste — one event, the way a terminal delivers pasted
    # text. Sent as keystrokes instead, a long prompt was still being consumed
    # when the following `tui-keys Enter` arrived, and was submitted cut short.
    _tmux("set-buffer", "-b", "midgectl", args.text)
    _tmux("paste-buffer", "-p", "-d", "-b", "midgectl", "-t", TMUX_SESSION)
    return 0


def tui_screen(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.timeout
    pattern = re.compile(args.until) if args.until else None
    while True:
        screen = _tmux("capture-pane", "-p", "-t", TMUX_SESSION)
        if pattern is None or pattern.search(screen) or time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    print(screen)
    if pattern is not None and not pattern.search(screen):
        print(f"harness: {args.until!r} not on screen after {args.timeout}s", file=sys.stderr)
        return 1
    return 0


def _tui_down() -> None:
    _tmux("kill-session", "-t", TMUX_SESSION, check=False)
    _docker("rm", "-f", TUI_CONTAINER, check=False)


def tui_down(_: argparse.Namespace) -> int:
    _tui_down()
    print(f"{TMUX_SESSION} removed")
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
    up_p.add_argument("midge_args", nargs=argparse.REMAINDER,
                      help="extra args passed to `midge --rpc` (put them last)")
    up_p.set_defaults(fn=up)

    for name, fn in (("down", down), ("logs", logs)):
        sub.add_parser(name).set_defaults(fn=fn)

    for name, fn, help_ in (
        ("tui", tui, "print the docker run line for an interactive TUI"),
        ("tui-up", tui_up, "run the TUI inside a detached tmux session"),
    ):
        tui_p = sub.add_parser(name, help=help_)
        tui_p.add_argument("--config", default="config.toml",
                           help="fixture config: config.toml or config-models.toml")
        tui_p.add_argument("-e", "--env", action="append", default=[], metavar="K=V")
        if name == "tui-up":
            tui_p.add_argument("--width", type=int, default=160)
            tui_p.add_argument("--height", type=int, default=48)
        tui_p.add_argument("midge_args", nargs=argparse.REMAINDER,
                           help="extra args passed to `midge` (put them last)")
        tui_p.set_defaults(fn=fn)

    keys_p = sub.add_parser("tui-keys", help="send tmux key names to the TUI")
    keys_p.add_argument("keys", nargs="+")
    keys_p.set_defaults(fn=tui_keys)

    type_p = sub.add_parser("tui-type", help="type literal text into the TUI")
    type_p.add_argument("text")
    type_p.set_defaults(fn=tui_type)

    screen_p = sub.add_parser("tui-screen", help="print the rendered TUI screen")
    screen_p.add_argument("--until", help="poll until this regex is on screen")
    screen_p.add_argument("--timeout", type=float, default=30.0)
    screen_p.set_defaults(fn=tui_screen)

    sub.add_parser("tui-down", help="end the tmux session and its container").set_defaults(
        fn=tui_down
    )

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
