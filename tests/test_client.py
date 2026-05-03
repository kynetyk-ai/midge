from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from types import SimpleNamespace
from typing import Any

import pytest

from pi.client import (
    Client,
    Done,
    Error,
    StreamEvent,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from pi.messages import TextContent, ToolCall, UserMessage


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def _tcd(
    *,
    index: int,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> Any:
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=function)


class _FakeStream:
    def __init__(self, chunks: Iterable[Any]) -> None:
        self._chunks = list(chunks)
        self._raise_after: int | None = None
        self._exc: BaseException | None = None

    def fail_after(self, n: int, exc: BaseException) -> None:
        self._raise_after = n
        self._exc = exc

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> Any:
        if self._raise_after is not None and self._raise_after <= 0 and self._exc:
            raise self._exc
        if not self._chunks:
            raise StopAsyncIteration
        if self._raise_after is not None:
            self._raise_after -= 1
        return self._chunks.pop(0)


def _install_fake_stream(client: Client, stream: _FakeStream) -> None:
    async def create(**kwargs: Any) -> _FakeStream:
        return stream

    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


async def _collect(it: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [ev async for ev in it]


async def test_simple_text_response() -> None:
    client = Client()
    _install_fake_stream(
        client,
        _FakeStream(
            [
                _chunk(content="Hello"),
                _chunk(content=", "),
                _chunk(content="world"),
                _chunk(finish_reason="stop"),
            ]
        ),
    )

    events = await _collect(
        client.stream(messages=[UserMessage(content="hi")], model="gpt-4o")
    )

    types = [type(e) for e in events]
    assert types == [StreamStart, TextStart, TextDelta, TextDelta, TextDelta, TextEnd, Done]
    assert isinstance(events[-1], Done)
    assert events[-1].message.stop_reason == "stop"
    assert isinstance(events[-1].message.content[0], TextContent)
    assert events[-1].message.content[0].text == "Hello, world"
    assert isinstance(events[5], TextEnd)
    assert events[5].content == "Hello, world"


async def test_tool_call_response() -> None:
    client = Client()
    _install_fake_stream(
        client,
        _FakeStream(
            [
                _chunk(tool_calls=[_tcd(index=0, id="call_1", name="read", arguments='{"path"')]),
                _chunk(tool_calls=[_tcd(index=0, arguments=': "/etc/hosts"}')]),
                _chunk(finish_reason="tool_calls"),
            ]
        ),
    )

    events = await _collect(
        client.stream(messages=[UserMessage(content="read it")], model="gpt-4o")
    )

    types = [type(e) for e in events]
    assert types == [StreamStart, ToolCallStart, ToolCallDelta, ToolCallDelta, ToolCallEnd, Done]

    end = events[-2]
    assert isinstance(end, ToolCallEnd)
    assert end.tool_call.id == "call_1"
    assert end.tool_call.name == "read"
    assert end.tool_call.arguments == {"path": "/etc/hosts"}

    done = events[-1]
    assert isinstance(done, Done)
    assert done.message.stop_reason == "tool_use"


async def test_text_then_tool_call() -> None:
    client = Client()
    _install_fake_stream(
        client,
        _FakeStream(
            [
                _chunk(content="reading "),
                _chunk(content="now"),
                _chunk(tool_calls=[_tcd(index=0, id="c1", name="read", arguments='{"path":"a"}')]),
                _chunk(finish_reason="tool_calls"),
            ]
        ),
    )

    events = await _collect(
        client.stream(messages=[UserMessage(content="x")], model="gpt-4o")
    )

    done = events[-1]
    assert isinstance(done, Done)
    msg = done.message
    assert len(msg.content) == 2
    assert isinstance(msg.content[0], TextContent)
    assert msg.content[0].text == "reading now"
    assert isinstance(msg.content[1], ToolCall)
    assert msg.content[1].arguments == {"path": "a"}


async def test_multiple_parallel_tool_calls() -> None:
    client = Client()
    _install_fake_stream(
        client,
        _FakeStream(
            [
                _chunk(tool_calls=[_tcd(index=0, id="c1", name="read", arguments='{"path":')]),
                _chunk(tool_calls=[_tcd(index=1, id="c2", name="read", arguments='{"path":')]),
                _chunk(tool_calls=[_tcd(index=0, arguments='"a"}')]),
                _chunk(tool_calls=[_tcd(index=1, arguments='"b"}')]),
                _chunk(finish_reason="tool_calls"),
            ]
        ),
    )

    events = await _collect(
        client.stream(messages=[UserMessage(content="x")], model="gpt-4o")
    )

    done = events[-1]
    assert isinstance(done, Done)
    tool_calls = [c for c in done.message.content if isinstance(c, ToolCall)]
    assert len(tool_calls) == 2
    assert tool_calls[0].id == "c1"
    assert tool_calls[0].arguments == {"path": "a"}
    assert tool_calls[1].id == "c2"
    assert tool_calls[1].arguments == {"path": "b"}


async def test_finish_reason_content_filter_maps_to_error() -> None:
    client = Client()
    _install_fake_stream(
        client,
        _FakeStream(
            [
                _chunk(content="part"),
                _chunk(finish_reason="content_filter"),
            ]
        ),
    )

    events = await _collect(
        client.stream(messages=[UserMessage(content="x")], model="gpt-4o")
    )

    done = events[-1]
    assert isinstance(done, Done)
    assert done.message.stop_reason == "error"


async def test_finish_reason_length_maps_to_length() -> None:
    client = Client()
    _install_fake_stream(
        client,
        _FakeStream([_chunk(content="x"), _chunk(finish_reason="length")]),
    )

    events = await _collect(
        client.stream(messages=[UserMessage(content="x")], model="gpt-4o")
    )

    done = events[-1]
    assert isinstance(done, Done)
    assert done.message.stop_reason == "length"


async def test_error_during_stream_emits_error_event() -> None:
    client = Client()
    fake = _FakeStream([_chunk(content="hello")])
    fake.fail_after(1, RuntimeError("network died"))
    _install_fake_stream(client, fake)

    events = await _collect(
        client.stream(messages=[UserMessage(content="x")], model="gpt-4o")
    )

    assert isinstance(events[-1], Error)
    assert events[-1].message.stop_reason == "error"
    assert events[-1].message.error_message == "network died"


async def test_cancelled_during_stream_emits_aborted_and_reraises() -> None:
    client = Client()
    fake = _FakeStream([_chunk(content="hello")])
    fake.fail_after(1, asyncio.CancelledError())
    _install_fake_stream(client, fake)

    collected: list[StreamEvent] = []
    with pytest.raises(asyncio.CancelledError):
        async for ev in client.stream(messages=[UserMessage(content="x")], model="gpt-4o"):
            collected.append(ev)

    assert isinstance(collected[-1], Error)
    assert collected[-1].message.stop_reason == "aborted"


async def test_partial_appears_in_every_event() -> None:
    client = Client()
    _install_fake_stream(
        client,
        _FakeStream(
            [
                _chunk(content="hi"),
                _chunk(finish_reason="stop"),
            ]
        ),
    )

    events = await _collect(
        client.stream(messages=[UserMessage(content="x")], model="gpt-4o")
    )

    partials = [
        getattr(e, "partial", None) or getattr(e, "message", None)
        for e in events
    ]
    first = partials[0]
    assert all(p is first for p in partials)


async def test_invalid_tool_call_arguments_become_empty_dict() -> None:
    client = Client()
    _install_fake_stream(
        client,
        _FakeStream(
            [
                _chunk(tool_calls=[_tcd(index=0, id="c1", name="x", arguments="not json")]),
                _chunk(finish_reason="tool_calls"),
            ]
        ),
    )

    events = await _collect(
        client.stream(messages=[UserMessage(content="x")], model="gpt-4o")
    )

    done = events[-1]
    assert isinstance(done, Done)
    tc = done.message.content[0]
    assert isinstance(tc, ToolCall)
    assert tc.arguments == {}
