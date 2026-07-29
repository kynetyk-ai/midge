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
import logging
import os
import sys
from pathlib import Path

from midge.agent import Agent
from midge.client import Client
from midge.extensions import BUILTIN_TOOL_DIRS, load_extensions
from midge.hooks import Hooks
from midge.logs import configure as configure_logging
from midge.logs import provider_host
from midge.rpc import RpcServer, claim_stdout
from midge.subagents import bind_subagents

# Not `__name__`: run as `-m`, that is "__main__", which sits outside the
# `midge` logger tree and so never picks up the configured level.
_logger = logging.getLogger("midge.examples.rpc_agent")

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
    # Before anything else can write: take fd 1 for the protocol and send
    # everything else — including a stray print() from a tool or extension —
    # to stderr.
    stdout = claim_stdout()
    configure_logging()
    # Without a Hooks object an extension's `register_hooks` never runs, so a
    # tool-approval policy loaded here would be silently inert — and a
    # sub-agent inherits whatever the parent has.
    hooks = Hooks()
    registry, prompt_addition = load_extensions(
        [*BUILTIN_TOOL_DIRS, *extension_dirs], hooks=hooks
    )
    full_prompt = BASE_SYSTEM_PROMPT
    if prompt_addition:
        full_prompt += "\n\n" + prompt_addition

    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("MIDGE_MODEL", "gpt-4o-mini")
    _logger.info(
        "startup mode=rpc model=%s provider=%s tools=%d",
        model,
        provider_host(base_url),
        len(registry),
    )
    client = Client(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url,
    )
    bind_subagents(registry, client=client, model=model, hooks=hooks)
    agent = Agent(
        client=client,
        model=model,
        tools=registry,
        system_prompt=full_prompt,
        hooks=hooks,
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
        stdout.write(data)
        stdout.flush()

    server = RpcServer(agent, session=None)
    await server.serve(read_line=read_line, write=write)
    return 0


def main() -> None:
    args = _parse_args(sys.argv[1:])
    sys.exit(asyncio.run(amain(list(args.extension_dir))))


if __name__ == "__main__":
    main()
