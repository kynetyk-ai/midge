"""Show `MidgeClient` in use.

Embedders import `midge.rpc.client` (standard library only) rather than copy this file.
"""

from __future__ import annotations

import argparse
import shlex
import sys

from midge.rpc.client import MidgeClient, ProtocolError


def default_server() -> list[str]:
    import shutil

    exe = shutil.which("midge")
    return [exe, "--rpc"] if exe else [sys.executable, "-m", "midge", "--rpc"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask midge a question via RPC.")
    parser.add_argument("prompt")
    parser.add_argument(
          "--server",
        type=shlex.split,
        default=None,
        help="command that starts the server (default: `midge --rpc`)",
     )
    args = parser.parse_args()
    server = args.server or default_server()
    try:
        with MidgeClient(server) as client:
            status = 0
            for frame in client.prompt(args.prompt):
                if frame.get("agent"):
                    continue
                kind = frame.get("type")
                if kind == "assistant_text_delta":
                    print(frame["delta"], end="", flush=True)
                elif kind == "tool_execution_start":
                    print(f"\n[{frame['name']}]", file=sys.stderr, flush=True)
                elif kind == "error" or (kind == "response" and not frame.get("success")):
                    print(f"\nerror: {frame.get('message') or frame.get('error')}", file=sys.stderr)
                    status = 1
            print()
            return status
    except ProtocolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
