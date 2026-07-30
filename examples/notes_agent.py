"""Notes-domain TUI — Phase 5 adaptability proof.

Same harness, same TUI, same agent loop — only the extensions and the system
prompt differ. No `BUILTIN_TOOL_DIRS` is loaded, so the model has access to the
notes tools and *only* the notes tools (no bash/edit/read/write).

Usage:
    poetry run python -m examples.notes_agent [--session PATH]

The KB lives at `~/.midge-notes/kb.json` by default; override with
`MIDGE_NOTES_KB`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from midge.agent import Agent
from midge.client import Client
from midge.config import Config
from midge.config import emit as emit_config_diagnostics
from midge.extensions import load_extensions
from midge.logs import configure as configure_logging
from midge.persistence import Session, resolve_session_path
from midge.tui import run_tui, tui_log_handler

NOTES_SYSTEM_PROMPT = (
    "You are a personal knowledge assistant. Help the user capture, find, "
    "and connect their notes. Prefer searching existing notes before "
    "answering from general knowledge — the user's own writing is more "
    "trustworthy than your training data for their work. When you learn "
    "something new from the user, suggest saving it as a note."
)

_EXTENSION_DIR = Path(__file__).parent / "notes_extension"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="examples.notes_agent")
    parser.add_argument(
        "--session",
        type=Path,
        metavar="PATH",
        help=(
            "Append turns to a JSONL session file. Resumes if the file exists. "
            "A relative path is taken under the session directory. Without this, "
            "a timestamped file is created there."
        ),
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="Do not write a transcript for this run.",
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
    config, diagnostics = Config.load()
    configure_logging(tui_log_handler(config.log.file), log=config.log)
    emit_config_diagnostics(diagnostics)

    registry, prompt_addition = load_extensions([_EXTENSION_DIR])
    full_prompt = NOTES_SYSTEM_PROMPT
    if prompt_addition:
        full_prompt += "\n\n" + prompt_addition

    session: Session | None = None
    model = config.model
    session_file = resolve_session_path(
        None if args.no_session else args.session,
        directory=config.session.dir,
        enabled=config.session.enabled and not args.no_session,
    )
    if session_file is not None:
        session = Session.open(session_file, model=model, system_prompt=full_prompt)
        model = session.model
        full_prompt = session.system_prompt or full_prompt

    client = Client(
        base_url=config.base_url,
        provider=config.provider,
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
