"""Notes-domain TUI — Phase 5 adaptability proof.

Same harness, same TUI, same agent loop — only the skills and the system
prompt differ. No `BUILTIN_DIRS` is loaded, so the model has access to the
notes tools and *only* the notes tools (no bash/edit/read/write).

Usage:
    poetry run python -m examples.notes_agent [--session PATH]

The KB lives at `~/.pym-notes/kb.json` by default; override with
`PYM_NOTES_KB`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pym.agent import Agent
from pym.client import Client
from pym.persistence import Session
from pym.skills import load_skills
from pym.tui import run_tui

NOTES_SYSTEM_PROMPT = (
    "You are a personal knowledge assistant. Help the user capture, find, "
    "and connect their notes. Prefer searching existing notes before "
    "answering from general knowledge — the user's own writing is more "
    "trustworthy than your training data for their work. When you learn "
    "something new from the user, suggest saving it as a note."
)

_SKILL_DIR = Path(__file__).parent / "notes_skill"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="examples.notes_agent")
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
        help="Run compaction after a turn if estimated history tokens exceed N.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    registry, prompt_addition = load_skills([_SKILL_DIR])
    full_prompt = NOTES_SYSTEM_PROMPT
    if prompt_addition:
        full_prompt += "\n\n" + prompt_addition

    session: Session | None = None
    if args.session is not None:
        if args.session.exists():
            session = Session.load(args.session)
            model = session.header.model
            full_prompt = session.header.system_prompt or full_prompt
        else:
            model = os.getenv("PI_MODEL", "gpt-4o-mini")
            session = Session.new(args.session, model=model, system_prompt=full_prompt)
    else:
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
        )
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main(sys.argv[1:])
