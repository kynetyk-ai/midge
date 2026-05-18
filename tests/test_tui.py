from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import pytest

from pi.agent import Agent
from pi.client import Client
from pi.tui.app import AssistantBubble, PiApp, UserBubble


def _chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


class _FakeStream:
    def __init__(self, chunks: Iterable[Any]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> Any:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _install_turns(client: Client, turns: list[list[Any]]) -> None:
    iterator = iter(turns)

    async def create(**kwargs: Any) -> _FakeStream:
        return _FakeStream(next(iterator))

    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def _build_agent(turns: list[list[Any]]) -> Agent:
    client = Client()
    _install_turns(client, turns)
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
        [[_chunk(content="hello"), _chunk(content=" there"), _chunk(finish_reason="stop")]]
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
        [[_chunk(content="ok"), _chunk(finish_reason="stop")]]
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
