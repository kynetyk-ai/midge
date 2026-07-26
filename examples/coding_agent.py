"""Run the coding agent from the command line.

Usage:
    poetry run python -m examples.coding_agent [options] "<prompt>"

Options:
    --skill-dir DIR       Add a skills directory (repeatable).
    --export-html PATH    Write HTML transcript on completion.
    --session PATH        Append turns to a JSONL session file. If the file
                          exists, the agent resumes from it (re-using the
                          model and system prompt from its header).

Env: OPENAI_API_KEY, OPENAI_BASE_URL, PYM_MODEL (default: gpt-4o-mini).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from pym.agent import Agent, AgentEnd, ToolExecutionEnd, ToolExecutionStart
from pym.client import Client, TextDelta
from pym.compaction import compact, needs_compaction
from pym.messages import TextContent
from pym.persistence import Session
from pym.session import export_html
from pym.skills import BUILTIN_DIRS, load_skills

BASE_SYSTEM_PROMPT = (
    "You are a coding assistant working in a local repository. "
    "Use the available tools to inspect and modify files. "
    "Keep responses concise and prefer doing work over describing it."
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="examples.coding_agent")
    parser.add_argument(
        "--skill-dir",
        action="append",
        type=Path,
        default=[],
        metavar="DIR",
        help="Directory of skill .py files to load (repeatable).",
    )
    parser.add_argument(
        "--export-html",
        type=Path,
        metavar="PATH",
        help="After the run completes, write a single-file HTML transcript to PATH.",
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
            "Disabled if not set. Estimator is bytes/4 — set generously."
        ),
    )
    parser.add_argument(
        "--compaction-keep-recent",
        type=int,
        default=20_000,
        metavar="N",
        help="Token budget for the suffix kept verbatim after compaction (default: 20000).",
    )
    parser.add_argument(
        "prompt",
        nargs="+",
        help="Prompt to send to the agent (joined with spaces).",
    )
    return parser.parse_args(argv)


async def amain(
    prompt: str,
    skill_dirs: list[Path],
    export_html_path: Path | None = None,
    session_path: Path | None = None,
    compaction_threshold: int | None = None,
    compaction_keep_recent: int = 20_000,
) -> int:
    registry, prompt_addition = load_skills([*BUILTIN_DIRS, *skill_dirs])

    session: Session | None = None
    if session_path is not None:
        if session_path.exists():
            session = Session.load(session_path)
            model = session.header.model
            full_prompt = session.header.system_prompt or ""
        else:
            full_prompt = BASE_SYSTEM_PROMPT
            if prompt_addition:
                full_prompt += "\n\n" + prompt_addition
            model = os.getenv("PYM_MODEL", "gpt-4o-mini")
            session = Session.new(
                session_path, model=model, system_prompt=full_prompt
            )
    else:
        full_prompt = BASE_SYSTEM_PROMPT
        if prompt_addition:
            full_prompt += "\n\n" + prompt_addition
        model = os.getenv("PYM_MODEL", "gpt-4o-mini")

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
        async for ev in agent.stream(prompt):
            if isinstance(ev, TextDelta):
                sys.stdout.write(ev.delta)
                sys.stdout.flush()
            elif isinstance(ev, ToolExecutionStart):
                sys.stdout.write(f"\n[{ev.tool_call.name}({ev.tool_call.arguments})]\n")
                sys.stdout.flush()
            elif isinstance(ev, ToolExecutionEnd):
                preview = ""
                if ev.result.content and isinstance(ev.result.content[0], TextContent):
                    preview = ev.result.content[0].text
                tag = "ERR" if ev.result.is_error else "OK"
                sys.stdout.write(f"  ↳ [{tag}] {preview[:200]}\n")
                sys.stdout.flush()
            elif isinstance(ev, AgentEnd):
                sys.stdout.write("\n")
                if session is not None:
                    session.append_many(ev.new_messages)

        if compaction_threshold is not None and needs_compaction(
            agent.history, threshold_tokens=compaction_threshold
        ):
            sys.stdout.write("[compacting context...]\n")
            sys.stdout.flush()
            result = await compact(
                agent.history,
                client=client,
                model=agent.model,
                keep_recent_tokens=compaction_keep_recent,
            )
            if result is not None:
                new_history, summary_text, cut_idx = result
                agent.history = new_history
                if session is not None:
                    session.append_compaction(summary=summary_text, cut_index=cut_idx)
                sys.stdout.write(
                    f"[compacted: {cut_idx} messages summarized; "
                    f"history is now {len(new_history)} messages]\n"
                )
    finally:
        if session is not None:
            session.close()

    if export_html_path is not None:
        export_html_path.write_text(
            export_html(
                agent.history,
                title=f"pym · {prompt[:60]}",
                model=agent.model,
            ),
            encoding="utf-8",
        )
        sys.stdout.write(f"[exported HTML transcript to {export_html_path}]\n")

    return 0


def main() -> None:
    args = _parse_args(sys.argv[1:])
    sys.exit(
        asyncio.run(
            amain(
                " ".join(args.prompt),
                list(args.skill_dir),
                export_html_path=args.export_html,
                session_path=args.session,
                compaction_threshold=args.compaction_threshold,
                compaction_keep_recent=args.compaction_keep_recent,
            )
        )
    )


if __name__ == "__main__":
    main()
