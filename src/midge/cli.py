"""`midge` CLI: launches the interactive TUI.

Usage:
    midge [--extension-dir DIR] [--skill-dir DIR] [--session PATH] \\
       [--compaction-threshold N] [--compaction-keep-recent N]

Env: OPENAI_API_KEY, OPENAI_BASE_URL, MIDGE_MODEL (default: gpt-4o-mini).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from midge.agent import Agent
from midge.client import Client
from midge.extensions import BUILTIN_TOOL_DIRS, load_extensions
from midge.hooks import Hooks, SessionEnd, SessionStart
from midge.logs import configure as configure_logging
from midge.logs import provider_host
from midge.persistence import Session
from midge.rpc import RpcServer, serve_stdio
from midge.skills import default_skill_dirs, load_skills, skills_prompt
from midge.subagents import bind_subagents
from midge.tui import run_tui, tui_log_handler

_logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = (
    "You are a coding assistant working in a local repository. "
    "Use the available tools to inspect and modify files. "
    "Keep responses concise and prefer doing work over describing it."
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="midge")
    parser.add_argument(
        "--extension-dir",
        action="append",
        type=Path,
        default=[],
        metavar="DIR",
        help="Directory of extension .py files to load (repeatable).",
    )
    parser.add_argument(
        "--skill-dir",
        action="append",
        type=Path,
        default=[],
        metavar="DIR",
        help=(
            "Directory of SKILL.md skills to load (repeatable). Searched before "
            "the project and user defaults."
        ),
    )
    parser.add_argument(
        "--rpc",
        action="store_true",
        help=(
            "Serve the JSON-over-stdio RPC protocol instead of the TUI. Stdout "
            "carries the protocol; diagnostics go to stderr."
        ),
    )
    parser.add_argument(
        "--session",
        type=Path,
        metavar="PATH",
        help="Append turns to a JSONL session file. Resumes if the file exists.",
    )
    parser.add_argument(
        "--compaction-threshold",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Run compaction after a turn if estimated history tokens exceed N. "
            "Disabled if not set."
        ),
    )
    parser.add_argument(
        "--compaction-keep-recent",
        type=int,
        default=20_000,
        metavar="N",
        help="Token budget for the suffix kept verbatim after compaction (default 20000).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    # Before `load_extensions`/`load_skills`, which are the two loudest
    # loaders — a handler installed after them loses every startup diagnostic.
    # The TUI needs a handler that resolves per record; RPC needs stderr,
    # because a handler bound to stdout would corrupt the protocol.
    configure_logging(None if args.rpc else tui_log_handler())
    hooks = Hooks()
    registry, prompt_addition = load_extensions(
        [*BUILTIN_TOOL_DIRS, *args.extension_dir], hooks=hooks
    )
    # Explicit paths outrank the defaults: naming a directory on the command
    # line is a deliberate override. Note this is the opposite nesting from the
    # extension dirs above, where the built-ins must not be shadowed.
    skills = load_skills([*args.skill_dir, *default_skill_dirs()])
    catalogue = skills_prompt(skills) if "read" in registry else ""

    session: Session | None = None
    model = os.getenv("MIDGE_MODEL", "gpt-4o-mini")
    durable = BASE_SYSTEM_PROMPT

    if args.session is not None:
        if args.session.exists():
            session = Session.load(args.session)
            model = session.header.model
            durable = session.header.system_prompt or BASE_SYSTEM_PROMPT
        else:
            session = Session.new(args.session, model=model, system_prompt=durable)

    # The header records the agent's identity. Which tools and skills exist is a
    # fact about this machine right now, so it is recomposed on every start
    # rather than restored — otherwise a skill added after the session began is
    # invisible, and its absolute paths could point at another machine entirely.
    full_prompt = "\n\n".join(p for p in (durable, prompt_addition, catalogue) if p)

    base_url = os.getenv("OPENAI_BASE_URL")
    _logger.info(
        "startup mode=%s model=%s provider=%s tools=%d skills=%d session=%s",
        "rpc" if args.rpc else "tui",
        model,
        provider_host(base_url),
        len(registry),
        len(skills),
        args.session or "-",
    )
    client = Client(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url,
    )
    # Tools cannot reach the calling agent, so any sub-agent tool an extension
    # registered gets what it needs to run a child here. No-op without them.
    bind_subagents(
        registry,
        client=client,
        model=model,
        hooks=hooks,
        session_path=args.session,
    )
    agent = Agent(
        client=client,
        model=model,
        tools=registry,
        system_prompt=full_prompt,
        hooks=hooks,
    )
    if session is not None:
        agent.history = list(session.messages)

    session_path = str(args.session) if args.session is not None else None

    if args.rpc:
        # RPC owns its loop, so the session bookends run inside it rather than
        # in their own `asyncio.run` the way the TUI's do.
        server = RpcServer(
            agent,
            session=session,
            compaction_keep_recent=args.compaction_keep_recent,
            base_prompt=durable,
            prompt_suffix="\n\n".join(p for p in (prompt_addition, catalogue) if p),
        )

        async def _serve() -> None:
            await hooks.emit(SessionStart(path=session_path))
            try:
                await serve_stdio(server)
            finally:
                if server.session is not None:
                    server.session.close()
                await hooks.emit(SessionEnd(path=session_path))

        asyncio.run(_serve())
        return

    # These bookend the TUI's own event loop, so they run in their own.
    # A handler that needs the running app's loop should use a turn-scoped
    # event instead.
    asyncio.run(hooks.emit(SessionStart(path=session_path)))
    try:
        run_tui(
            agent,
            session=session,
            compaction_threshold=args.compaction_threshold,
            compaction_keep_recent=args.compaction_keep_recent,
        )
    finally:
        if session is not None:
            session.close()
        asyncio.run(hooks.emit(SessionEnd(path=session_path)))
