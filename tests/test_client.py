"""The streaming state machine and the retry policy — provider-independent.

Driven by `FakeProvider`, so these tests say nothing about any wire format.
Chunk parsing lives in `test_providers_openai.py`; the last section here runs a
few SDK-shaped chunks through the real adapter so the seam between `decode` and
this state machine stays exercised.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
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
)
from midge.messages import Message, TextContent, ToolCall, UserMessage
from midge.providers.openai_compat import OpenAIProvider
from tests.fakes import (
    FakeProvider,
    finish,
    install,
    install_provider,
    say,
    tcall,
    tokens,
)

USER: list[Message] = [UserMessage(content="x")]


async def _collect(it: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [ev async for ev in it]


async def _run(client: Client) -> list[StreamEvent]:
    return await _collect(client.stream(messages=USER, model="gpt-4o"))


def _status_error(status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "http://x")
    response = httpx.Response(status_code=status, request=request)
    return openai.APIStatusError("boom", response=response, body=None)


# --- assembling a response ------------------------------------------------


async def test_simple_text_response() -> None:
    client = Client()
    install(client, [[say("hel"), say("lo"), finish()]])

    events = await _run(client)

    assert [type(e) for e in events] == [
        StreamStart,
        TextStart,
        TextDelta,
        TextDelta,
        TextEnd,
        Done,
    ]
    done = events[-1]
    assert isinstance(done, Done)
    assert isinstance(done.message.content[0], TextContent)
    assert done.message.content[0].text == "hello"
    assert done.message.stop_reason == "stop"


async def test_tool_call_response() -> None:
    client = Client()
    install(
        client,
        [
            [
                tcall(index=0, id="c1", name="read", args='{"path"'),
                tcall(index=0, args=': "/etc/hosts"}'),
                finish("tool_use"),
            ]
        ],
    )

    events = await _run(client)

    ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert len(ends) == 1
    assert ends[0].tool_call.name == "read"
    assert ends[0].tool_call.arguments == {"path": "/etc/hosts"}
    assert any(isinstance(e, ToolCallStart) for e in events)
    assert any(isinstance(e, ToolCallDelta) for e in events)


async def test_text_then_tool_call_keeps_both_blocks() -> None:
    client = Client()
    install(
        client,
        [
            [
                say("thinking"),
                tcall(index=0, id="c1", name="read", args='{"path":"a"}'),
                finish("tool_use"),
            ]
        ],
    )

    done = (await _run(client))[-1]
    assert isinstance(done, Done)
    assert isinstance(done.message.content[0], TextContent)
    assert isinstance(done.message.content[1], ToolCall)


async def test_parallel_tool_calls_are_tracked_by_index() -> None:
    client = Client()
    install(
        client,
        [
            [
                tcall(index=0, id="c1", name="read", args='{"path":'),
                tcall(index=1, id="c2", name="read", args='{"path":'),
                tcall(index=0, args='"a"}'),
                tcall(index=1, args='"b"}'),
                finish("tool_use"),
            ]
        ],
    )

    done = (await _run(client))[-1]
    assert isinstance(done, Done)
    calls = [c for c in done.message.content if isinstance(c, ToolCall)]
    assert [c.id for c in calls] == ["c1", "c2"]
    assert [c.arguments for c in calls] == [{"path": "a"}, {"path": "b"}]


async def test_a_missing_id_falls_back_to_the_index() -> None:
    client = Client()
    install(client, [[tcall(index=3, name="read", args="{}"), finish("tool_use")]])

    done = (await _run(client))[-1]
    assert isinstance(done, Done)
    assert isinstance(done.message.content[0], ToolCall)
    assert done.message.content[0].id == "call_3"


async def test_a_stream_with_no_stop_reason_still_stops() -> None:
    client = Client()
    install(client, [[say("hi")]])

    done = (await _run(client))[-1]
    assert isinstance(done, Done)
    assert done.message.stop_reason == "stop"


async def test_invalid_tool_call_arguments_become_empty_dict() -> None:
    # Executing a call whose arguments are unknown is worse than failing it, so
    # the parse error is recorded rather than guessed at.
    client = Client()
    install(client, [[tcall(index=0, id="c1", name="x", args="not json"), finish("tool_use")]])

    done = (await _run(client))[-1]
    assert isinstance(done, Done)
    tc = done.message.content[0]
    assert isinstance(tc, ToolCall)
    assert tc.arguments == {}
    assert tc.arguments_error is not None


async def test_partial_is_the_same_object_in_every_event() -> None:
    # Consumers hold the reference handed out by StreamStart, so it must never be
    # rebound.
    client = Client()
    install(client, [[say("a"), finish()]])

    events = await _run(client)
    partials = {id(getattr(e, "partial", None)) for e in events if hasattr(e, "partial")}
    assert len(partials) == 1


# --- usage ----------------------------------------------------------------


async def test_usage_is_captured() -> None:
    client = Client()
    install(client, [[say("hi"), finish(), tokens(input=1200, output=35, cached=1024)]])

    done = (await _run(client))[-1]
    assert isinstance(done, Done)
    assert done.message.usage is not None
    assert (done.message.usage.input, done.message.usage.cached) == (1200, 1024)


async def test_a_usage_only_chunk_does_not_disturb_content() -> None:
    client = Client()
    install(client, [[say("hi"), tokens(input=5, output=1), finish()]])

    done = (await _run(client))[-1]
    assert isinstance(done, Done)
    assert len(done.message.content) == 1
    assert isinstance(done.message.content[0], TextContent)
    assert done.message.content[0].text == "hi"


async def test_usage_from_a_failed_attempt_does_not_survive_a_retry() -> None:
    client = Client(retry_base_delay=0)
    install(
        client,
        [
            [tokens(input=999, output=999), _status_error(503)],
            [say("ok"), finish()],
        ],
    )

    done = (await _run(client))[-1]
    assert isinstance(done, Done)
    assert done.message.usage is None


# --- failure and cancellation ---------------------------------------------


async def test_error_during_stream_emits_an_error_event() -> None:
    client = Client()
    install(client, [[RuntimeError("upstream blew up")]])

    last = (await _run(client))[-1]
    assert isinstance(last, Error)
    assert last.message.stop_reason == "error"
    assert "upstream blew up" in (last.message.error_message or "")


async def test_cancellation_emits_aborted_and_reraises() -> None:
    client = Client()

    class _Hang(FakeProvider):
        async def open(self, body: dict[str, Any]) -> AsyncIterator[Any]:
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

    client.provider = _Hang()
    collected: list[StreamEvent] = []

    async def run() -> None:
        async for ev in client.stream(messages=USER, model="gpt-4o"):
            collected.append(ev)

    task = asyncio.create_task(run())
    for _ in range(10):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(1):
            await task

    last = collected[-1]
    assert isinstance(last, Error)
    assert last.message.stop_reason == "aborted"


# --- retry policy ---------------------------------------------------------


async def test_retries_then_succeeds() -> None:
    client = Client(retry_base_delay=0)
    provider = install_provider(client, [[_status_error(503)], [say("second"), finish()]])

    done = (await _run(client))[-1]

    assert provider.attempts == 2
    assert isinstance(done, Done)
    assert isinstance(done.message.content[0], TextContent)
    assert done.message.content[0].text == "second"


async def test_retry_exhausted_emits_error() -> None:
    client = Client(retry_base_delay=0, max_attempts=2)
    provider = install_provider(client, [[_status_error(503)], [_status_error(503)]])

    assert isinstance((await _run(client))[-1], Error)
    assert provider.attempts == 2


async def test_non_retryable_fails_on_the_first_attempt() -> None:
    client = Client(retry_base_delay=0)
    provider = install_provider(client, [[_status_error(400)]])

    assert isinstance((await _run(client))[-1], Error)
    assert provider.attempts == 1


async def test_no_retry_once_content_has_been_yielded() -> None:
    # A mid-stream drop after deltas escaped cannot be replayed: the consumer has
    # already seen part of the response.
    client = Client(retry_base_delay=0)
    provider = install_provider(
        client,
        [
            [say("hello"), _status_error(503)],
            [say("second"), finish()],
        ],
    )

    events = await _run(client)

    assert provider.attempts == 1
    last = events[-1]
    assert isinstance(last, Error)
    assert isinstance(last.message.content[0], TextContent)
    assert last.message.content[0].text == "hello"


async def test_retry_discards_partial_state_from_the_failed_attempt() -> None:
    # The failed attempt buffered nothing that escaped, so the retry starts from
    # an empty message rather than inheriting it.
    client = Client(retry_base_delay=0)
    install(client, [[_status_error(503)], [say("clean"), finish()]])

    done = (await _run(client))[-1]
    assert isinstance(done, Done)
    assert len(done.message.content) == 1
    assert isinstance(done.message.content[0], TextContent)
    assert done.message.content[0].text == "clean"


async def test_cancellation_during_retry_backoff_propagates() -> None:
    # The whole reason the backoff sleep is ours rather than the SDK's: the SDK's
    # sleep ignores cancellation, so Ctrl+C during it would do nothing.
    client = Client(retry_base_delay=3600)
    provider = install_provider(client, [[_status_error(503)], [_status_error(503)]])
    collected: list[StreamEvent] = []

    async def run() -> None:
        async for ev in client.stream(messages=USER, model="gpt-4o"):
            collected.append(ev)

    task = asyncio.create_task(run())
    for _ in range(10):
        await asyncio.sleep(0)

    # Parked in the backoff: the first attempt is spent, the second not started.
    assert provider.attempts == 1
    assert not task.done()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(1):
            await task

    assert provider.attempts == 1
    assert [type(e) for e in collected] == [StreamStart]


# --- provider selection ---------------------------------------------------


def test_the_resolved_provider_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    # The base_url heuristic is a guess, so a misrouted request must not be a
    # mystery.
    with caplog.at_level(logging.INFO, logger="midge.client"):
        Client(api_key="k", base_url="http://localhost:11434/v1")
    assert any(
        "provider_selected name=openai-compatible" in r.getMessage() for r in caplog.records
    )


def test_an_explicit_provider_instance_is_used_as_is() -> None:
    provider = FakeProvider()
    assert Client(provider=provider).provider is provider


# --- end to end through the real adapter ----------------------------------
#
# Everything above fakes the transport. These run SDK-shaped chunks through the
# real `decode`, so a regression in the seam between the adapter and the state
# machine cannot hide behind the fake.


def _sdk_chunk(*, content: str | None = None, finish_reason: str | None = None) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)], usage=None
    )


def _sdk_tool_chunk(*, index: int, id: str, name: str, arguments: str) -> Any:
    tcd = SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )
    delta = SimpleNamespace(content=None, tool_calls=[tcd])
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None
    )


def _real_provider_yielding(chunks: list[Any]) -> OpenAIProvider:
    provider = OpenAIProvider(api_key="test")

    async def create(**kwargs: Any) -> Any:
        async def gen() -> AsyncIterator[Any]:
            for c in chunks:
                yield c

        return gen()

    provider._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return provider


async def test_real_adapter_text_reaches_the_state_machine() -> None:
    client = Client(
        provider=_real_provider_yielding(
            [
                _sdk_chunk(content="hel"),
                _sdk_chunk(content="lo"),
                _sdk_chunk(finish_reason="stop"),
            ]
        )
    )

    done = (await _run(client))[-1]
    assert isinstance(done, Done)
    assert isinstance(done.message.content[0], TextContent)
    assert done.message.content[0].text == "hello"
    assert done.message.stop_reason == "stop"


async def test_real_adapter_tool_call_reaches_the_state_machine() -> None:
    client = Client(
        provider=_real_provider_yielding(
            [
                _sdk_tool_chunk(index=0, id="c1", name="read", arguments='{"path":'),
                _sdk_tool_chunk(index=0, id="c1", name="read", arguments='"a"}'),
                _sdk_chunk(finish_reason="tool_calls"),
            ]
        )
    )

    done = (await _run(client))[-1]
    assert isinstance(done, Done)
    tc = done.message.content[0]
    assert isinstance(tc, ToolCall)
    assert tc.arguments == {"path": "a"}
    assert done.message.stop_reason == "tool_use"
