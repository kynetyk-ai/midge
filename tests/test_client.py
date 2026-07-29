from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from midge.client import (
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
    is_retryable,
)
from midge.messages import TextContent, ToolCall, UserMessage


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


def _install_attempts(client: Client, attempts: list[Any]) -> list[int]:
    """Install one outcome per attempt. An outcome is either a `_FakeStream`
    to return or an exception to raise from `create`. Returns a single-element
    list holding the call count."""
    calls = [0]
    queue = list(attempts)

    async def create(**kwargs: Any) -> _FakeStream:
        calls[0] += 1
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return calls


def _status_error(status: int) -> openai.APIStatusError:
    return openai.APIStatusError(
        f"status {status}",
        response=httpx.Response(
            status_code=status, request=httpx.Request("POST", "http://x/v1")
        ),
        body=None,
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
    assert events[-1].message.error_message == "RuntimeError: network died"


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


def test_is_retryable_classification() -> None:
    assert is_retryable(_status_error(429))
    assert is_retryable(_status_error(500))
    assert is_retryable(_status_error(503))
    assert is_retryable(openai.APIConnectionError(request=httpx.Request("POST", "http://x")))
    assert is_retryable(openai.APITimeoutError(request=httpx.Request("POST", "http://x")))

    assert not is_retryable(_status_error(400))
    assert not is_retryable(_status_error(401))
    assert not is_retryable(_status_error(404))
    assert not is_retryable(RuntimeError("boom"))


async def test_sdk_retries_disabled() -> None:
    # The SDK's own backoff sleep ignores cancellation, so the retry must be ours.
    assert Client()._client.max_retries == 0


async def test_retries_then_succeeds() -> None:
    client = Client(retry_base_delay=0)
    calls = _install_attempts(
        client,
        [
            _status_error(503),
            _status_error(429),
            _FakeStream([_chunk(content="hi"), _chunk(finish_reason="stop")]),
        ],
    )

    events = await _collect(
        client.stream(messages=[UserMessage(content="x")], model="gpt-4o")
    )

    assert calls[0] == 3
    done = events[-1]
    assert isinstance(done, Done)
    assert done.message.stop_reason == "stop"
    assert isinstance(done.message.content[0], TextContent)
    assert done.message.content[0].text == "hi"
    # Exactly one StreamStart despite three attempts.
    assert [type(e) for e in events].count(StreamStart) == 1


async def test_retry_exhausted_emits_error() -> None:
    client = Client(max_attempts=3, retry_base_delay=0)
    calls = _install_attempts(client, [_status_error(500)] * 3)

    events = await _collect(
        client.stream(messages=[UserMessage(content="x")], model="gpt-4o")
    )

    assert calls[0] == 3
    assert isinstance(events[-1], Error)
    assert events[-1].message.stop_reason == "error"


async def test_non_retryable_fails_on_first_attempt() -> None:
    client = Client(retry_base_delay=0)
    calls = _install_attempts(client, [_status_error(401)])

    events = await _collect(
        client.stream(messages=[UserMessage(content="x")], model="gpt-4o")
    )

    assert calls[0] == 1
    assert isinstance(events[-1], Error)


async def test_no_retry_once_content_has_been_yielded() -> None:
    # A mid-stream drop after deltas escaped cannot be replayed: the consumer
    # has already seen part of the response.
    client = Client(retry_base_delay=0)
    failing = _FakeStream([_chunk(content="hello")])
    failing.fail_after(1, _status_error(503))
    calls = _install_attempts(
        client,
        [failing, _FakeStream([_chunk(content="second"), _chunk(finish_reason="stop")])],
    )

    events = await _collect(
        client.stream(messages=[UserMessage(content="x")], model="gpt-4o")
    )

    assert calls[0] == 1
    assert isinstance(events[-1], Error)
    assert isinstance(events[-1].message.content[0], TextContent)
    assert events[-1].message.content[0].text == "hello"


async def test_retry_discards_partial_state_from_failed_attempt() -> None:
    # The first attempt buffers a tool call but dies before any event escapes,
    # so the retry must not inherit it.
    class _DropBeforeYield(_FakeStream):
        async def __anext__(self) -> Any:
            raise _status_error(503)

    client = Client(retry_base_delay=0)
    _install_attempts(
        client,
        [
            _DropBeforeYield([]),
            _FakeStream(
                [
                    _chunk(tool_calls=[_tcd(index=0, id="c1", name="read", arguments='{"path":"a"}')]),
                    _chunk(finish_reason="tool_calls"),
                ]
            ),
        ],
    )

    events = await _collect(
        client.stream(messages=[UserMessage(content="x")], model="gpt-4o")
    )

    done = events[-1]
    assert isinstance(done, Done)
    assert len(done.message.content) == 1
    tc = done.message.content[0]
    assert isinstance(tc, ToolCall)
    assert tc.id == "c1"
    # Still the object handed out by StreamStart.
    start = events[0]
    assert isinstance(start, StreamStart)
    assert done.message is start.partial


async def test_cancellation_during_retry_backoff_propagates() -> None:
    # The whole reason the backoff sleep is ours rather than the SDK's: the
    # SDK's sleep ignores cancellation, so Ctrl+C during it would do nothing.
    client = Client(retry_base_delay=3600)
    calls = _install_attempts(client, [_status_error(503), _status_error(503)])
    collected: list[StreamEvent] = []

    async def run() -> None:
        async for ev in client.stream(messages=[UserMessage(content="x")], model="gpt-4o"):
            collected.append(ev)

    task = asyncio.create_task(run())
    for _ in range(10):
        await asyncio.sleep(0)

    # Parked in the backoff: the first attempt is spent, the second not started.
    assert calls[0] == 1
    assert not task.done()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(1):
            await task

    assert calls[0] == 1
    # Cancelled between attempts, so nothing beyond StreamStart was emitted.
    assert [type(e) for e in collected] == [StreamStart]


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


# ---- token usage ----


def _usage_chunk(*, prompt: int, completion: int, cached: int = 0) -> Any:
    """The provider's final chunk: usage present, `choices` empty."""
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )


async def test_usage_is_captured_from_the_final_chunk() -> None:
    client = Client()
    _install_fake_stream(
        client,
        _FakeStream(
            [
                _chunk(content="hi"),
                _chunk(finish_reason="stop"),
                _usage_chunk(prompt=1200, completion=35, cached=1024),
            ]
        ),
    )

    events = await _collect(client.stream(messages=[UserMessage(content="hi")], model="m"))
    done = events[-1]
    assert isinstance(done, Done)
    assert done.message.usage is not None
    assert done.message.usage.input == 1200
    assert done.message.usage.output == 35
    assert done.message.usage.cached == 1024


async def test_usage_chunk_does_not_disturb_content() -> None:
    client = Client()
    _install_fake_stream(
        client,
        _FakeStream(
            [
                _chunk(content="a"),
                _usage_chunk(prompt=5, completion=1),
                _chunk(content="b"),
                _chunk(finish_reason="stop"),
            ]
        ),
    )

    events = await _collect(client.stream(messages=[UserMessage(content="hi")], model="m"))
    done = events[-1]
    assert isinstance(done, Done)
    assert isinstance(done.message.content[0], TextContent)
    assert done.message.content[0].text == "ab"


async def test_stream_options_requested_by_default() -> None:
    client = Client()
    captured: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> _FakeStream:
        captured.append(kwargs)
        return _FakeStream([_chunk(finish_reason="stop")])

    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    await _collect(client.stream(messages=[UserMessage(content="hi")], model="m"))

    assert captured[0]["stream_options"] == {"include_usage": True}


async def test_stream_options_can_be_switched_off() -> None:
    # Some OpenAI-compatible servers 400 on stream_options, which would fail
    # the whole turn.
    client = Client(include_usage=False)
    captured: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> _FakeStream:
        captured.append(kwargs)
        return _FakeStream([_chunk(finish_reason="stop")])

    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    await _collect(client.stream(messages=[UserMessage(content="hi")], model="m"))

    assert "stream_options" not in captured[0]


async def test_include_usage_env_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIDGE_INCLUDE_USAGE", "0")
    assert Client()._include_usage is False
    monkeypatch.setenv("MIDGE_INCLUDE_USAGE", "1")
    assert Client()._include_usage is True


async def test_usage_absent_when_the_server_omits_it() -> None:
    client = Client()
    _install_fake_stream(
        client, _FakeStream([_chunk(content="hi"), _chunk(finish_reason="stop")])
    )

    events = await _collect(client.stream(messages=[UserMessage(content="hi")], model="m"))
    done = events[-1]
    assert isinstance(done, Done)
    assert done.message.usage is None


class _FailingStream:
    """Emits chunks, then raises — a provider dying after reporting usage."""

    def __init__(self, chunks: list[Any], error: BaseException) -> None:
        self._chunks = list(chunks)
        self._error = error

    def __aiter__(self) -> _FailingStream:
        return self

    async def __anext__(self) -> Any:
        if self._chunks:
            return self._chunks.pop(0)
        raise self._error


async def test_usage_from_a_failed_attempt_does_not_survive_a_retry() -> None:
    # Each attempt resets per-attempt state; usage counted against a request
    # that then failed must not be attributed to the one that succeeded.
    client = Client(retry_base_delay=0)
    _install_attempts(
        client,
        [
            _FailingStream([_usage_chunk(prompt=999, completion=999)], _status_error(429)),
            _FakeStream([_chunk(content="ok"), _chunk(finish_reason="stop")]),
        ],
    )

    events = await _collect(client.stream(messages=[UserMessage(content="hi")], model="m"))
    done = events[-1]
    assert isinstance(done, Done)
    assert done.message.usage is None
