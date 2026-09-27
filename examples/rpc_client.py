"""A minimal client for `midge --rpc`: the shape to copy when embedding midge.

    python examples/rpc_client.py "read README.md and say what it is for"
    python examples/rpc_client.py --server "midge --rpc --profile reviewer" "review the diff"

Standard library only, on purpose: a program that embeds midge talks to it over
a pipe, and should not need midge's dependencies to do so. The protocol it
speaks is `docs/rpc.md`; this uses the part every client needs:

1. spawn the server and read its first frame, `ready`;
2. refuse a protocol version it was not written for;
3. send a `prompt` and render events until `agent_settled` — always the last
   frame of a prompt, whether it succeeded, failed or was aborted;
4. close stdin, which is how a client says it is done.

Everything else — steering, aborting, sessions, profiles — is more commands on
the same pipe.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from typing import Any

SUPPORTED_PROTOCOLS = {1}


def default_server() -> list[str]:
    exe = shutil.which("midge")
    return [exe, "--rpc"] if exe else [sys.executable, "-m", "midge", "--rpc"]


def run(server: list[str], prompt: str) -> int:
    proc = subprocess.Popen(server, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert proc.stdin is not None and proc.stdout is not None

    def frames() -> Any:
        for line in proc.stdout:  # type: ignore[union-attr]
            yield json.loads(line)

    stream = frames()
    ready = next(stream, None)
    if ready is None or ready.get("type") != "ready":
        print(f"error: expected a `ready` frame, got {ready!r}", file=sys.stderr)
        return 2
    if ready.get("protocol") not in SUPPORTED_PROTOCOLS:
        print(
            f"error: midge speaks protocol {ready.get('protocol')}; this client knows "
            f"{sorted(SUPPORTED_PROTOCOLS)}",
            file=sys.stderr,
        )
        proc.stdin.close()
        proc.wait()
        return 2

    proc.stdin.write(json.dumps({"id": "1", "type": "prompt", "message": prompt}) + "\n")
    proc.stdin.flush()

    status = 0
    for frame in stream:
        kind = frame.get("type")
        if frame.get("agent"):
            continue  # a sub-agent's activity; its own transcript has the detail
        if kind == "assistant_text_delta":
            print(frame["delta"], end="", flush=True)
        elif kind == "tool_execution_start":
            print(f"\n[{frame['name']}]", file=sys.stderr, flush=True)
        elif kind == "error" or (kind == "response" and not frame.get("success")):
            print(f"\nerror: {frame.get('message') or frame.get('error')}", file=sys.stderr)
            status = 1
        elif kind == "agent_settled":
            break
        # Anything else — including frame types added after this was written —
        # is ignored, as the compatibility policy asks of every client.

    print()
    proc.stdin.close()
    return proc.wait() or status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prompt")
    parser.add_argument(
        "--server",
        type=shlex.split,
        default=None,
        help="command that starts the server (default: `midge --rpc`)",
    )
    args = parser.parse_args()
    return run(args.server or default_server(), args.prompt)


if __name__ == "__main__":
    sys.exit(main())
