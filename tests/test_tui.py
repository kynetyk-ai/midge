from __future__ import annotations

import asyncio
from typing import Any

import pytest
from textual.widgets import TextArea

from midge.agent import Agent
from midge.client import Client
from midge.commands import Controls
from midge.config import ProviderConfig
from midge.messages import UserMessage
from midge.providers import ModelRegistry
from midge.tui.app import AssistantBubble, MidgeCommands, PiApp, StatusLine, UserBubble
from tests.fakes import finish, install, install_gated, say


def _build_agent(turns: list[list[Any]]) -> Agent:
    client = Client()
    install(client, turns)
    return Agent(client=client, model="m")


def _app(turns: list[list[Any]], **kw: Any) -> PiApp:
    return PiApp(Controls(_build_agent(turns), **kw))


@pytest.mark.asyncio
async def test_app_launches_and_exits_cleanly() -> None:
    agent = _build_agent([])
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.is_running
        await pilot.press("ctrl+d")
    # If we got here without hanging, the app exited cleanly.


@pytest.mark.asyncio
async def test_submit_creates_user_and_assistant_bubbles() -> None:
    agent = _build_agent(
        [[say("hello"), say(" there"), finish()]]
    )
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        # Type into the input area and submit via Enter
        input_widget = app.query_one("#input")
        input_widget.text = "hi"  # type: ignore[attr-defined]
        await pilot.press("enter")

        # Wait for the run worker to complete
        await app.workers.wait_for_complete()
        await pilot.pause()

        user_bubbles = list(app.query(UserBubble))
        assistant_bubbles = list(app.query(AssistantBubble))
        assert len(user_bubbles) == 1
        assert len(assistant_bubbles) == 1
        # The assistant bubble accumulated the streamed deltas
        assert assistant_bubbles[0]._text == "hello there"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_escape_clears_input() -> None:
    agent = _build_agent([])
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input")
        input_widget.text = "draft text"  # type: ignore[attr-defined]
        await pilot.press("escape")
        assert input_widget.text == ""  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_alt_enter_inserts_newline_without_submitting() -> None:
    agent = _build_agent([])
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input")
        input_widget.text = "line one"  # type: ignore[attr-defined]
        await pilot.press("alt+enter")
        await pilot.pause()

        # Newline got inserted; nothing was submitted
        assert "\n" in input_widget.text  # type: ignore[attr-defined]
        assert len(list(app.query(UserBubble))) == 0


@pytest.mark.asyncio
async def test_enter_on_empty_input_is_noop() -> None:
    agent = _build_agent([])
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert len(list(app.query(UserBubble))) == 0


@pytest.mark.asyncio
async def test_multiline_prompt_submits_via_enter_with_newlines_intact() -> None:
    agent = _build_agent(
        [[say("ok"), finish()]]
    )
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input")
        input_widget.text = "line one\nline two"  # type: ignore[attr-defined]
        await pilot.press("enter")

        await app.workers.wait_for_complete()
        await pilot.pause()

        user_bubbles = list(app.query(UserBubble))
        assert len(user_bubbles) == 1
        # The agent's history should carry both lines
        history = agent.history
        assert len(history) >= 1
        first = history[0]
        assert hasattr(first, "content")
        assert "line one" in str(first.content)
        assert "line two" in str(first.content)


# --- the command surface --------------------------------------------------


def _status(app: PiApp) -> list[str]:
    return [str(s.visual) for s in app.query(StatusLine)]


async def _settle(pilot: Any) -> None:
    """`_on_submit` is async and mounts its result, so one pause is not enough."""
    await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
async def test_a_slash_command_runs_instead_of_prompting() -> None:
    agent = _build_agent([])
    agent.history.extend([UserMessage(content="a"), UserMessage(content="b")])
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "/clear_context"
        await pilot.press("enter")
        await _settle(pilot)

        assert agent.history == []
        assert any("cleared 2 messages" in s for s in _status(app))
        assert not list(app.query(UserBubble))


@pytest.mark.asyncio
async def test_a_message_that_merely_starts_with_a_slash_is_a_prompt() -> None:
    # The reason only known names intercept: this is a sentence about a path,
    # and rejecting it as an unknown command would make the input box refuse
    # ordinary English.
    app = _app([[say("ok"), finish()]])
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "/etc/hosts is missing"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert [str(b.visual) for b in app.query(UserBubble)] == ["/etc/hosts is missing"]


@pytest.mark.asyncio
async def test_a_command_argument_reaches_the_operation() -> None:
    app = _app([])
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "/set_model gpt-4o"
        await pilot.press("enter")
        await _settle(pilot)

        assert app.agent.model == "gpt-4o"
        assert any("model is now gpt-4o" in s for s in _status(app))


@pytest.mark.asyncio
async def test_a_refusal_is_shown_rather_than_raised() -> None:
    app = _app([])
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "/set_session_name whatever"
        await pilot.press("enter")
        await _settle(pilot)

        assert any("no session" in s for s in _status(app))
        assert app.is_running


@pytest.mark.asyncio
async def test_the_palette_offers_every_argument_free_builtin() -> None:
    app = _app([])
    async with app.run_test() as pilot:
        await pilot.pause()
        offered = {d for d, _desc, _arg in MidgeCommands(app.screen)._entries()}

        # Free-text commands are absent: a palette entry has nowhere for a path.
        assert {"abort", "compact", "clear_context", "reload"} <= offered
        assert not any(o.startswith("new_session") for o in offered)
        # And `set_model` only once a `[models]` table says what the values are
        # — offering it bare would fire it with no model at all.
        assert not any(o.startswith("set_model") for o in offered)


@pytest.mark.asyncio
async def test_a_model_registry_becomes_palette_entries() -> None:
    # `builtin_schema` narrows `set_model` to the registered ids, so the palette
    # offers them as choices rather than asking for free text.
    registry = ModelRegistry(
        models={"a-model": "p", "b-model": "p"},
        providers={"p": ProviderConfig(kind="openai")},
    )
    agent = Agent(client=Client(registry=registry), model="a-model")
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        await pilot.pause()
        offered = {d for d, _desc, _arg in MidgeCommands(app.screen)._entries()}

        assert {"set_model a-model", "set_model b-model"} <= offered


@pytest.mark.asyncio
async def test_typing_mid_turn_queues_rather_than_cancelling() -> None:
    # The old path cancelled the running worker and awaited its teardown so two
    # writers could not interleave into `Agent.history`. Steering removes the
    # second writer instead, and keeps the work already done.
    gate = asyncio.Event()
    client = Client()
    provider = install_gated(client, [say("first"), finish()], gate)
    assert provider is not None
    agent = Agent(client=client, model="m")
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "start"
        await pilot.press("enter")
        await _settle(pilot)
        assert app.busy()

        app.query_one("#input", TextArea).text = "and also this"
        await pilot.press("enter")
        await _settle(pilot)

        assert app.busy(), "the turn must still be running"
        assert agent.steering is not None and agent.steering.pending()
        assert any("queued: and also this" in s for s in _status(app))

        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()


@pytest.mark.asyncio
async def test_compact_is_refused_while_a_turn_is_running() -> None:
    # Compaction awaits a provider call before reassigning `agent.history`, so a
    # turn running across that window has its appended messages dropped.
    gate = asyncio.Event()
    client = Client()
    install_gated(client, [say("first"), finish()], gate)
    agent = Agent(client=client, model="m")
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "start"
        await pilot.press("enter")
        await _settle(pilot)

        await app.invoke("compact", None)
        await _settle(pilot)
        assert any("run is in flight" in s for s in _status(app))

        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_command_needing_an_argument_refuses_without_one() -> None:
    app = _app([])
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "/set_model"
        await pilot.press("enter")
        await _settle(pilot)

        assert app.agent.model == "m", "an empty argument must not be applied"
        assert any("needs an argument" in s for s in _status(app))
