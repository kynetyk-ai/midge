from __future__ import annotations

from typing import Any

import pytest

from midge.agent import Agent
from midge.client import Client
from midge.tui.app import AssistantBubble, PiApp, UserBubble
from tests.fakes import finish, install, say


def _build_agent(turns: list[list[Any]]) -> Agent:
    client = Client()
    install(client, turns)
    return Agent(client=client, model="m")


@pytest.mark.asyncio
async def test_app_launches_and_exits_cleanly() -> None:
    agent = _build_agent([])
    app = PiApp(agent)
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
    app = PiApp(agent)
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
    app = PiApp(agent)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input")
        input_widget.text = "draft text"  # type: ignore[attr-defined]
        await pilot.press("escape")
        assert input_widget.text == ""  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_alt_enter_inserts_newline_without_submitting() -> None:
    agent = _build_agent([])
    app = PiApp(agent)
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
    app = PiApp(agent)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert len(list(app.query(UserBubble))) == 0


@pytest.mark.asyncio
async def test_multiline_prompt_submits_via_enter_with_newlines_intact() -> None:
    agent = _build_agent(
        [[say("ok"), finish()]]
    )
    app = PiApp(agent)
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
