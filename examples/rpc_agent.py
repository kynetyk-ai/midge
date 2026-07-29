"""Run the agent in RPC (JSON-over-stdio) mode.

Reads newline-delimited JSON commands from stdin; writes responses and async
events to stdout. See `notes/rpc.md` and `src/midge/rpc.py` for the protocol.

Usage:
    poetry run python -m examples.rpc_agent [--extension-dir DIR ...]

Same env vars as `examples.coding_agent`: OPENAI_API_KEY, OPENAI_BASE_URL,
MIDGE_MODEL.

Try it from a separate shell:
    echo '{"id": "1", "type": "prompt", "message": "say hi"}' \\
      | poetry run python -m examples.rpc_agent
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from midge.agent import Agent
from midge.client import Client
from midge.extensions import BUILTIN_TOOL_DIRS, load_extensions
from midge.logs import configure as configure_logging
from midge.rpc import RpcServer

_READ_LIMIT = 16 * 1024 * 1024

BASE_SYSTEM_PROMPT = (
    "You are a coding assistant. "
    "Use the available tools to inspect and modify files. Keep responses concise."
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="examples.rpc_agent")
    parser.add_argument(
        "--extension-dir",
        action="append",
        type=Path,
        default=[],
        metavar="DIR",
        help="Directory of extension .py files to load (repeatable).",
    )
    return parser.parse_args(argv)


async def amain(extension_dirs: list[Path]) -> int:
    # Default handler: stderr. Stdout is the protocol here, so nothing may ever
    # be written to it but framed JSON.
    configure_logging()
    registry, prompt_addition = load_extensions([*BUILTIN_TOOL_DIRS, *extension_dirs])
    full_prompt = BASE_SYSTEM_PROMPT
    if prompt_addition:
        full_prompt += "\n\n" + prompt_addition

    client = Client(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    agent = Agent(
        client=client,
        model=os.getenv("MIDGE_MODEL", "gpt-4o-mini"),
        tools=registry,
        system_prompt=full_prompt,
    )

    loop = asyncio.get_running_loop()
    # The default 64 KiB limit turns a large pasted prompt into a
    # ValueError that escapes `serve()` and kills the server.
    reader = asyncio.StreamReader(limit=_READ_LIMIT)
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    async def read_line() -> bytes:
        return await reader.readline()

    async def write(data: bytes) -> None:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

    server = RpcServer(agent)
    await server.serve(read_line=read_line, write=write)
    return 0


def main() -> None:
    args = _parse_args(sys.argv[1:])
    sys.exit(asyncio.run(amain(list(args.extension_dir))))


if __name__ == "__main__":
    main()
