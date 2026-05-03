"""Run the agent in RPC (JSON-over-stdio) mode.

Reads newline-delimited JSON commands from stdin; writes responses and async
events to stdout. See `notes/rpc.md` and `src/pi/rpc.py` for the protocol.

Usage:
    poetry run python -m examples.rpc_agent [--skill-dir DIR ...]

Same env vars as `examples.coding_agent`: OPENAI_API_KEY, OPENAI_BASE_URL,
PI_MODEL.

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

from pi.agent import Agent
from pi.client import Client
from pi.rpc import RpcServer
from pi.skills import BUILTIN_DIRS, load_skills

BASE_SYSTEM_PROMPT = (
    "You are a coding assistant. "
    "Use the available tools to inspect and modify files. Keep responses concise."
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="examples.rpc_agent")
    parser.add_argument(
        "--skill-dir",
        action="append",
        type=Path,
        default=[],
        metavar="DIR",
        help="Directory of skill .py files to load (repeatable).",
    )
    return parser.parse_args(argv)


async def amain(skill_dirs: list[Path]) -> int:
    registry, prompt_addition = load_skills([*BUILTIN_DIRS, *skill_dirs])
    full_prompt = BASE_SYSTEM_PROMPT
    if prompt_addition:
        full_prompt += "\n\n" + prompt_addition

    client = Client(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    agent = Agent(
        client=client,
        model=os.getenv("PI_MODEL", "gpt-4o-mini"),
        tools=registry,
        system_prompt=full_prompt,
    )

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
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
    sys.exit(asyncio.run(amain(list(args.skill_dir))))


if __name__ == "__main__":
    main()
