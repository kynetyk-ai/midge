"""Run the coding agent from the command line.

Usage:
    poetry run python -m examples.coding_agent [options] "<prompt>"

Options:
    --extension-dir DIR   Add an extensions directory (repeatable).
    --skill-dir DIR       Add a SKILL.md skills directory (repeatable).
    --skill NAME          Force a skill; the prompt becomes its instructions.
    --export-html PATH    Write HTML transcript on completion.
    --session PATH        Append turns to a JSONL session file. If the file
                          exists, the agent resumes from it, re-using the model
                          from its header. Tool and skill availability is
                          recomposed from disk rather than restored.

Env: OPENAI_API_KEY, OPENAI_BASE_URL, MIDGE_MODEL (default: gpt-4o-mini).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from midge.agent import Agent, AgentEnd, ToolExecutionEnd, ToolExecutionStart
from midge.client import Client, TextDelta
from midge.compaction import compact, needs_compaction
from midge.extensions import BUILTIN_TOOL_DIRS, load_extensions
from midge.messages import TextContent, UserMessage
from midge.persistence import Session, TranscriptEntry, read_transcript
from midge.session import export_html
from midge.skills import default_skill_dirs, load_skills, skill_message, skills_prompt

BASE_SYSTEM_PROMPT = (
    "You are a coding assistant working in a local repository. "
    "Use the available tools to inspect and modify files. "
    "Keep responses concise and prefer doing work over describing it."
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="examples.coding_agent")
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
        "--skill",
        metavar="NAME",
        help=(
            "Force a skill instead of leaving the choice to the model: its body "
            "is sent as the turn, with the prompt appended as instructions."
        ),
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
    extension_dirs: list[Path],
    skill_dirs: list[Path] | None = None,
    force_skill: str | None = None,
    export_html_path: Path | None = None,
    session_path: Path | None = None,
    compaction_threshold: int | None = None,
    compaction_keep_recent: int = 20_000,
) -> int:
    registry, prompt_addition = load_extensions([*BUILTIN_TOOL_DIRS, *extension_dirs])
    skills = load_skills([*(skill_dirs or []), *default_skill_dirs()])
    catalogue = skills_prompt(skills) if "read" in registry else ""

    user_input: str | UserMessage = prompt
    if force_skill is not None:
        try:
            user_input = skill_message(skills, force_skill, prompt)
        except KeyError:
            available = ", ".join(sorted(s.name for s in skills)) or "none"
            sys.stderr.write(f"No skill named {force_skill!r}. Available: {available}\n")
            return 2

    session: Session | None = None
    model = os.getenv("MIDGE_MODEL", "gpt-4o-mini")
    durable = BASE_SYSTEM_PROMPT

    if session_path is not None:
        if session_path.exists():
            session = Session.load(session_path)
            model = session.header.model
            durable = session.header.system_prompt or BASE_SYSTEM_PROMPT
        else:
            session = Session.new(session_path, model=model, system_prompt=durable)

    # See cli.py: tool and skill availability is recomposed every start rather
    # than restored from the header, which froze at session creation.
    full_prompt = "\n\n".join(p for p in (durable, prompt_addition, catalogue) if p)

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

    # `new_messages` never reaches us if the turn is interrupted, so persist the
    # partial turn from the history tail instead.
    mark = len(agent.history)
    interrupted = False

    try:
        async for ev in agent.stream(user_input):
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
            try:
                result = await compact(
                    agent.history,
                    client=client,
                    model=agent.model,
                    keep_recent_tokens=compaction_keep_recent,
                )
            except Exception as e:
                # An otherwise successful turn should not exit non-zero because
                # the follow-on summarization call failed.
                sys.stdout.write(f"[compaction failed: {e}]\n")
                result = None
            if result is not None:
                new_history, summary_text, cut_idx = result
                agent.history = new_history
                if session is not None:
                    session.append_compaction(summary=summary_text, cut_index=cut_idx)
                sys.stdout.write(
                    f"[compacted: {cut_idx} messages summarized; "
                    f"history is now {len(new_history)} messages]\n"
                )
    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        if session is not None:
            session.append_many(agent.history[mark:])
        sys.stdout.write("\n[interrupted]\n")
    finally:
        if session is not None:
            session.close()

        if export_html_path is not None:
            # `agent.history` is post-compaction; the session file still holds
            # every message, so export from it whenever there is one.
            entries: Sequence[TranscriptEntry] = agent.history
            if session is not None:
                entries = read_transcript(session.path)[1]
            export_html_path.write_text(
                export_html(
                    entries,
                    title=f"midge · {prompt[:60]}",
                    model=agent.model,
                ),
                encoding="utf-8",
            )
            sys.stdout.write(f"[exported HTML transcript to {export_html_path}]\n")

    return 130 if interrupted else 0


def main() -> None:
    args = _parse_args(sys.argv[1:])
    # `asyncio.run` cancels the task on Ctrl+C (handled inside `amain`) and then
    # re-raises; swallow it here so the CLI exits without a traceback.
    try:
        code = asyncio.run(
            amain(
                " ".join(args.prompt),
                list(args.extension_dir),
                skill_dirs=list(args.skill_dir),
                force_skill=args.skill,
                export_html_path=args.export_html,
                session_path=args.session,
                compaction_threshold=args.compaction_threshold,
                compaction_keep_recent=args.compaction_keep_recent,
            )
        )
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()
