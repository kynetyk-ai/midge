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
import sys
from pathlib import Path

from midge.agent import Agent
from midge.client import Client
from midge.config import Config
from midge.config import emit as emit_config_diagnostics
from midge.extensions import BUILTIN_TOOL_DIRS, load_extensions
from midge.hooks import Hooks
from midge.logs import configure as configure_logging
from midge.logs import provider_host
from midge.rpc import RpcServer, serve_stdio
from midge.subagents import bind_subagents

# Not `__name__`: run as `-m`, that is "__main__", which sits outside the
# `midge` logger tree and so never picks up the configured level.
_logger = logging.getLogger("midge.examples.rpc_agent")

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
    config, diagnostics = Config.load()
    configure_logging(log=config.log)
    emit_config_diagnostics(diagnostics)
    # Without a Hooks object an extension's `register_hooks` never runs, so a
    # tool-approval policy loaded here would be silently inert — and a
    # sub-agent inherits whatever the parent has.
    hooks = Hooks()
    registry, prompt_addition = load_extensions([*BUILTIN_TOOL_DIRS, *extension_dirs], hooks=hooks)
    full_prompt = "\n\n".join(p for p in (BASE_SYSTEM_PROMPT, prompt_addition) if p)

    model = config.model
    _logger.info(
        "startup mode=rpc model=%s provider=%s tools=%d",
        model,
        provider_host(config.base_url),
        len(registry),
    )
    client = Client(
        base_url=config.base_url,
        provider=config.provider,
        max_attempts=config.retry.max_attempts,
        retry_base_delay=config.retry.base_delay,
    )
    bind_subagents(registry, client=client, model=model, hooks=hooks)
    agent = Agent(
        client=client,
        model=model,
        tools=registry,
        system_prompt=full_prompt,
        hooks=hooks,
    )

    # The halves are passed apart so `set_system_prompt` can change the base
    # without deleting what the extensions contributed.
    server = RpcServer(
        agent,
        session=None,
        base_prompt=BASE_SYSTEM_PROMPT,
        extension_prompt=prompt_addition,
    )
    # Claims stdout for the protocol, installs SIGTERM/SIGHUP handlers, and
    # reads stdin to EOF.
    await serve_stdio(server)
    return 0


def main() -> None:
    args = _parse_args(sys.argv[1:])
    sys.exit(asyncio.run(amain(list(args.extension_dir))))


if __name__ == "__main__":
    main()
