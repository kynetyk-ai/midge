"""`Controls.run_turn` — the one place a turn is persisted and compacted.

Both front-ends iterate it, so these tests drive it directly; `test_rpc` and
`test_tui` check that each front-end actually goes through it.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from pathlib import Path
from typing import Any

import pytest

from midge.agent import Agent, AgentEnd
from midge.client import Client, TextDelta
from midge.commands import CompactionEnd, CompactionStart, Controls
from midge.compaction import count_tokens
from midge.messages import AssistantMessage, Message, TextContent, UserMessage
from midge.persistence import Session, read_transcript
from midge.tools import ToolRegistry, tool
from tests.fakes import ScriptedProvider, finish, install, install_gated, say, tcall

# Small enough that everything but the last exchange is summarized.
_KEEP_ONE_TURN = 120


def _long_history() -> list[Message]:
    history: list[Message] = []
    for i in range(3):
        history.append(UserMessage(content=f"q{i}: " + "x" * 200))
        history.append(AssistantMessage(content=[TextContent(text="a" * 200)]))
    return history


def _controls(tmp_path: Path, client: Client, **kw: Any) -> tuple[Controls, Path]:
    path = tmp_path / "t.jsonl"
    agent = Agent(client=client, model="m")
    return Controls(agent, session=Session.new(path, model="m"), **kw), path


async def _drain(controls: Controls, message: str) -> list[Any]:
    async with aclosing(controls.run_turn(message)) as events:
        return [ev async for ev in events]


def _compactions(path: Path) -> list[Any]:
    _, entries = read_transcript(path)
    return [e for e in entries if getattr(e, "type", None) == "compaction"]


async def test_a_completed_turn_is_written_once(tmp_path: Path) -> None:
    client = Client()
    install(client, [[say("hi"), finish()]])
    controls, path = _controls(tmp_path, client)

    await _drain(controls, "ping")
    assert controls.session is not None
    controls.session.close()

    restored = Session.load(path).messages
    assert [m.role for m in restored] == ["user", "assistant"]


async def test_a_front_end_that_breaks_mid_render_still_persists_the_turn(
    tmp_path: Path,
) -> None:
    # #110's second hole: a render error used to skip persistence entirely,
    # because only `CancelledError` was caught.
    client = Client()
    install(client, [[say("hi"), finish()]])
    controls, path = _controls(tmp_path, client)

    with pytest.raises(RuntimeError, match="render"):
        async with aclosing(controls.run_turn("ping")) as events:
            async for ev in events:
                if isinstance(ev, TextDelta):
                    raise RuntimeError("render failed")

    restored = Session.load(path).messages
    assert restored and isinstance(restored[0], UserMessage)
    assert restored[0].content == "ping"


async def test_a_cancelled_turn_persists_what_it_had(tmp_path: Path) -> None:
    client = Client()
    gate = asyncio.Event()
    install_gated(client, [say("partial")], gate)
    controls, path = _controls(tmp_path, client)

    started = asyncio.Event()

    async def run() -> None:
        async with aclosing(controls.run_turn("ping")) as events:
            async for ev in events:
                if isinstance(ev, TextDelta):
                    started.set()

    task = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    restored = Session.load(path).messages
    assert restored[0].content == "ping"


async def test_no_threshold_means_no_compaction(tmp_path: Path) -> None:
    client = Client()
    install(client, [[say("hi"), finish()]])
    controls, _ = _controls(tmp_path, client)
    controls.agent.history = _long_history()

    events = await _drain(controls, "ping")
    assert not any(isinstance(e, CompactionStart | CompactionEnd) for e in events)


async def test_crossing_the_threshold_compacts_after_the_turn(tmp_path: Path) -> None:
    client = Client()
    install(client, [[say("hi"), finish()], [say("## Goal\nbe brief"), finish()]])
    history = _long_history()
    controls, path = _controls(
        tmp_path,
        client,
        compaction_threshold=1,
        compaction_keep_recent=count_tokens(history[-2:]) + 5,
    )
    controls.agent.history = history

    events = await _drain(controls, "ping")

    kinds = [type(e) for e in events]
    assert kinds[-2:] == [CompactionStart, CompactionEnd]
    end = events[-1]
    assert end.error is None and end.cut_index is not None
    assert end.message_count == len(controls.agent.history)
    assert len(_compactions(path)) == 1


async def test_a_failed_summary_leaves_history_alone(tmp_path: Path) -> None:
    client = Client()
    # The summary call gets an empty answer, which `summarize` refuses.
    install(client, [[say("hi"), finish()], [say(""), finish()]])
    controls, path = _controls(
        tmp_path, client, compaction_threshold=1, compaction_keep_recent=_KEEP_ONE_TURN
    )
    controls.agent.history = _long_history()

    events = await _drain(controls, "ping")

    end = events[-1]
    assert isinstance(end, CompactionEnd)
    assert end.error is not None
    assert len(controls.agent.history) == 6 + 2
    assert _compactions(path) == []


async def test_cancelling_during_compaction_changes_nothing(tmp_path: Path) -> None:
    # #114: the transcript and the in-memory history must agree afterwards —
    # neither compacted, rather than one of them.
    calls = 0
    never = asyncio.Event()

    async def on_open(body: dict[str, Any]) -> list[Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [say("hi"), finish()]
        await never.wait()
        return []

    client = Client()
    client.provider = ScriptedProvider(on_open)
    controls, path = _controls(
        tmp_path, client, compaction_threshold=1, compaction_keep_recent=_KEEP_ONE_TURN
    )
    controls.agent.history = _long_history()

    compacting = asyncio.Event()

    async def run() -> None:
        async with aclosing(controls.run_turn("ping")) as events:
            async for ev in events:
                if isinstance(ev, CompactionStart):
                    compacting.set()

    task = asyncio.create_task(run())
    await asyncio.wait_for(compacting.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(controls.agent.history) == 6 + 2
    assert _compactions(path) == []
    assert controls.session is not None
    controls.session.close()
    assert len(Session.load(path).messages) == 2


@tool
async def _noop() -> str:
    """Return a long result so the context grows."""
    return "ok " + "y" * 200


def _long_turn_replies() -> list[list]:
    """Six model responses for a long turn."""
    return [
        [tcall(id="c1", name="_noop", args="{}"), finish("tool_use")],
        [say("## Goal\nsummary one"), finish()],
        [tcall(id="c2", name="_noop", args="{}"), finish("tool_use")],
        [say("## Goal\nsummary two"), finish()],
        [say("done"), finish()],
        [say("## Goal\nsummary three"), finish()],
    ]


def _controls_with_tool(
    tmp_path: Path, client: Client, **kw: Any
) -> tuple[Controls, Path]:
    """Controls helper with `_noop` tool and pre-loaded session history."""
    path = tmp_path / "t.jsonl"
    history = _long_history()
    session = Session.new(path, model="m")
    session.append_many(history)
    keep_recent = count_tokens(history[-2:]) + 5
    agent = Agent(client=client, model="m", tools=ToolRegistry([_noop]))
    controls = Controls(
        agent, session=session, compaction_threshold=1, compaction_keep_recent=keep_recent, **kw
    )
    agent.history = list(history)
    return controls, path


async def test_compaction_runs_inside_a_long_turn(tmp_path: Path) -> None:
    client = Client()
    install(
        client,
        _long_turn_replies(),
    )
    controls, _ = _controls_with_tool(tmp_path, client)

    events = await _drain(controls, "go")

    first_compaction_end_idx = next(i for i, e in enumerate(events) if isinstance(e, CompactionEnd))
    agent_end_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentEnd))
    assert first_compaction_end_idx < agent_end_idx
    first_end = events[first_compaction_end_idx]
    assert first_end.cut_index is not None
    assert first_end.error is None


async def test_the_request_after_a_mid_turn_compaction_is_smaller(tmp_path: Path) -> None:
    client = Client()
    bodies = install(
        client,
        _long_turn_replies(),
    )
    controls, _ = _controls_with_tool(tmp_path, client)

    await _drain(controls, "go")
    # request 0 is before compaction, 1 is the summary, 2 is after compaction
    assert len(bodies[2]["messages"]) < len(bodies[0]["messages"])


async def test_a_session_replays_to_the_agents_history_after_mid_turn_compaction(
    tmp_path: Path,
) -> None:
    client = Client()
    install(
        client,
        _long_turn_replies(),
    )
    controls, path = _controls_with_tool(tmp_path, client)

    await _drain(controls, "go")
    assert controls.session is not None
    controls.session.close()

    assert Session.load(path).messages == controls.agent.history


async def test_abandoning_a_turn_after_a_mid_turn_compaction_leaves_a_replayable_session(
    tmp_path: Path,
) -> None:
    client = Client()
    install(
        client,
        _long_turn_replies(),
    )
    controls, path = _controls_with_tool(tmp_path, client)

    async with aclosing(controls.run_turn("go")) as events:
        async for ev in events:
            if isinstance(ev, CompactionEnd):
                break

    assert controls.session is not None
    controls.session.close()
    assert Session.load(path).messages == controls.agent.history
