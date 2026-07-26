"""`pym` CLI: launches the interactive TUI.

Usage:
    py [--skill-dir DIR] [--session PATH] [--compaction-threshold N] \\
       [--compaction-keep-recent N]

Env: OPENAI_API_KEY, OPENAI_BASE_URL, PI_MODEL (default: gpt-4o-mini).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pym.agent import Agent
from pym.client import Client
from pym.persistence import Session
from pym.skills import BUILTIN_DIRS, load_skills
from pym.tui import run_tui

BASE_SYSTEM_PROMPT = (
    "You are a coding assistant working in a local repository. "
    "Use the available tools to inspect and modify files. "
    "Keep responses concise and prefer doing work over describing it."
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pym")
    parser.add_argument(
        "--skill-dir",
        action="append",
        type=Path,
        default=[],
        metavar="DIR",
        help="Directory of skill .py files to load (repeatable).",
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
    registry, prompt_addition = load_skills([*BUILTIN_DIRS, *args.skill_dir])

    session: Session | None = None
    if args.session is not None:
        if args.session.exists():
            session = Session.load(args.session)
            model = session.header.model
            full_prompt = session.header.system_prompt or BASE_SYSTEM_PROMPT
        else:
            full_prompt = BASE_SYSTEM_PROMPT
            if prompt_addition:
                full_prompt += "\n\n" + prompt_addition
            model = os.getenv("PI_MODEL", "gpt-4o-mini")
            session = Session.new(args.session, model=model, system_prompt=full_prompt)
    else:
        full_prompt = BASE_SYSTEM_PROMPT
        if prompt_addition:
            full_prompt += "\n\n" + prompt_addition
        model = os.getenv("PI_MODEL", "gpt-4o-mini")

    client = Client(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    agent = Agent(
        client=client,
        model=model,
        tools=registry,
        system_prompt=full_prompt,
    )
    if session is not None:
        agent.history = list(session.messages)

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
