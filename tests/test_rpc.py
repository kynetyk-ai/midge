from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

from pym.agent import Agent
from pym.client import Client
from pym.rpc import RpcServer, event_to_wire
from pym.tools import ToolRegistry, tool


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


class _SlowStream:
    """Stream that pauses on a sentinel so a test can observe in-flight state."""

    def __init__(self, chunks_before: list[Any], gate: asyncio.Event) -> None:
        self._before = list(chunks_before)
        self._gate = gate

    def __aiter__(self) -> _SlowStream:
        return self

    async def __anext__(self) -> Any:
        if self._before:
            return self._before.pop(0)
        await self._gate.wait()
        raise StopAsyncIteration


def _install_slow_stream(client: Client, before: list[Any], gate: asyncio.Event) -> None:
    async def create(**kwargs: Any) -> _SlowStream:
        return _SlowStream(before, gate)

    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


class _Inbox:
    def __init__(self) -> None:
        self._q: asyncio.Queue[bytes] = asyncio.Queue()

    async def feed(self, line: bytes) -> None:
        await self._q.put(line)

    async def feed_text(self, line: str) -> None:
        await self._q.put(line.encode("utf-8"))

    def close(self) -> None:
        self._q.put_nowait(b"")

    async def read_line(self) -> bytes:
        return await self._q.get()


class _Outbox:
    def __init__(self) -> None:
        self.lines: list[dict[str, Any]] = []
        self._buffer = b""

    async def write(self, data: bytes) -> None:
        self._buffer += data
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            if line:
                self.lines.append(json.loads(line.decode("utf-8")))


async def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise TimeoutError("predicate did not become true in time")
        await asyncio.sleep(0.005)


def _start_server(agent: Agent) -> tuple[RpcServer, _Inbox, _Outbox, asyncio.Task[None]]:
    server = RpcServer(agent)
    inbox = _Inbox()
    outbox = _Outbox()
    task = asyncio.create_task(
        server.serve(read_line=inbox.read_line, write=outbox.write)
    )
    return server, inbox, outbox, task


async def test_prompt_response_then_events() -> None:
    client = Client()
    _install_turns(client, [[_chunk(content="hi"), _chunk(finish_reason="stop")]])
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "prompt", "message": "ping"}\n')
    await _wait_for(lambda: any(line.get("type") == "agent_end" for line in outbox.lines))
    inbox.close()
    await task

    types = [line.get("type") for line in outbox.lines]
    assert types[0] == "response"
    assert outbox.lines[0]["id"] == "r1"
    assert outbox.lines[0]["success"] is True
    assert "user_message" in types
    assert "assistant_text_delta" in types
    assert "assistant_message_end" in types
    assert "agent_end" in types


async def test_get_messages_returns_history() -> None:
    client = Client()
    _install_turns(client, [[_chunk(content="hi"), _chunk(finish_reason="stop")]])
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "prompt", "message": "ping"}\n')
    await _wait_for(lambda: any(line.get("type") == "agent_end" for line in outbox.lines))
    await inbox.feed_text('{"id": "r2", "type": "get_messages"}\n')
    await _wait_for(
        lambda: any(
            line.get("type") == "response" and line.get("id") == "r2"
            for line in outbox.lines
        )
    )
    inbox.close()
    await task

    resp = next(line for line in outbox.lines if line.get("id") == "r2")
    assert resp["success"] is True
    data = resp["data"]
    assert len(data) == 2
    assert data[0]["role"] == "user"
    assert data[1]["role"] == "assistant"


async def test_abort_cancels_in_flight_prompt() -> None:
    client = Client()
    gate = asyncio.Event()
    _install_slow_stream(client, [_chunk(content="part")], gate)
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "prompt", "message": "go"}\n')
    await _wait_for(
        lambda: any(line.get("type") == "assistant_text_delta" for line in outbox.lines)
    )

    await inbox.feed_text('{"id": "r2", "type": "abort"}\n')
    await _wait_for(
        lambda: any(
            line.get("type") == "response" and line.get("id") == "r2"
            for line in outbox.lines
        )
    )
    await _wait_for(
        lambda: any(
            line.get("type") == "error" and line.get("stop_reason") == "aborted"
            for line in outbox.lines
        )
    )

    gate.set()
    inbox.close()
    await task

    abort_resp = next(line for line in outbox.lines if line.get("id") == "r2")
    assert abort_resp["success"] is True
    error_events = [
        line for line in outbox.lines
        if line.get("type") == "error" and line.get("stop_reason") == "aborted"
    ]
    assert len(error_events) == 1, "expected exactly one cancelled-error event"


async def test_abort_when_idle_responds_failure() -> None:
    client = Client()
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "abort"}\n')
    await _wait_for(lambda: any(line.get("id") == "r1" for line in outbox.lines))
    inbox.close()
    await task

    resp = outbox.lines[0]
    assert resp["success"] is False
    assert "no prompt in flight" in resp["error"]


async def test_parse_error_responds_with_command_parse() -> None:
    client = Client()
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text("not valid json\n")
    await _wait_for(lambda: any(line.get("command") == "parse" for line in outbox.lines))
    inbox.close()
    await task

    resp = outbox.lines[0]
    assert resp["type"] == "response"
    assert resp["command"] == "parse"
    assert resp["success"] is False
    assert "id" not in resp


async def test_unknown_command_responds_failure() -> None:
    client = Client()
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "no_such_command"}\n')
    await _wait_for(lambda: any(line.get("id") == "r1" for line in outbox.lines))
    inbox.close()
    await task

    resp = outbox.lines[0]
    assert resp["success"] is False
    assert "unknown command" in resp["error"]


async def test_prompt_missing_message_responds_failure() -> None:
    client = Client()
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "prompt"}\n')
    await _wait_for(lambda: any(line.get("id") == "r1" for line in outbox.lines))
    inbox.close()
    await task

    resp = outbox.lines[0]
    assert resp["success"] is False
    assert "message" in resp["error"]


async def test_prompt_already_in_flight_rejected() -> None:
    client = Client()
    gate = asyncio.Event()
    _install_slow_stream(client, [_chunk(content="x")], gate)
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "prompt", "message": "first"}\n')
    await _wait_for(
        lambda: any(line.get("type") == "assistant_text_delta" for line in outbox.lines)
    )
    await inbox.feed_text('{"id": "r2", "type": "prompt", "message": "second"}\n')
    await _wait_for(lambda: any(line.get("id") == "r2" for line in outbox.lines))

    gate.set()
    inbox.close()
    await task

    resp = next(line for line in outbox.lines if line.get("id") == "r2")
    assert resp["success"] is False
    assert "already in flight" in resp["error"]


async def test_no_id_means_no_id_in_response() -> None:
    client = Client()
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"type": "abort"}\n')
    await _wait_for(lambda: any(line.get("type") == "response" for line in outbox.lines))
    inbox.close()
    await task

    resp = outbox.lines[0]
    assert "id" not in resp


async def test_tool_call_events_flow_through() -> None:
    @tool
    async def echo(text: str) -> str:
        return f"echoed:{text}"

    client = Client()
    _install_turns(
        client,
        [
            [
                _chunk(
                    tool_calls=[_tcd(index=0, id="c1", name="echo", arguments='{"text":"hi"}')]
                ),
                _chunk(finish_reason="tool_calls"),
            ],
            [_chunk(content="ok"), _chunk(finish_reason="stop")],
        ],
    )
    agent = Agent(client=client, model="m", tools=ToolRegistry([echo]))
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "prompt", "message": "go"}\n')
    await _wait_for(lambda: any(line.get("type") == "agent_end" for line in outbox.lines))
    inbox.close()
    await task

    types = [line.get("type") for line in outbox.lines]
    assert "tool_call_start" in types
    assert "tool_call_end" in types
    assert "tool_execution_start" in types
    assert "tool_result" in types

    tool_result = next(line for line in outbox.lines if line.get("type") == "tool_result")
    assert tool_result["tool_call_id"] == "c1"
    assert tool_result["content"] == "echoed:hi"
    assert tool_result["is_error"] is False


def test_event_to_wire_drops_internal_events() -> None:
    from pym.client import StreamStart, TextEnd, TextStart
    from pym.messages import AssistantMessage

    partial = AssistantMessage()
    assert event_to_wire(StreamStart(partial=partial)) is None
    assert event_to_wire(TextStart(content_index=0, partial=partial)) is None
    assert event_to_wire(TextEnd(content_index=0, content="x", partial=partial)) is None


def test_event_to_wire_text_delta() -> None:
    from pym.client import TextDelta
    from pym.messages import AssistantMessage, TextContent

    partial = AssistantMessage(content=[TextContent(text="hel")])
    wire = event_to_wire(TextDelta(content_index=0, delta="lo", partial=partial))
    assert wire == {"type": "assistant_text_delta", "delta": "lo"}


def test_event_to_wire_tool_call_end() -> None:
    from pym.client import ToolCallEnd
    from pym.messages import AssistantMessage, ToolCall

    tc = ToolCall(id="c1", name="read", arguments={"path": "x"})
    partial = AssistantMessage(content=[tc])
    wire = event_to_wire(
        ToolCallEnd(content_index=0, tool_call=tc, partial=partial)
    )
    assert wire == {
        "type": "tool_call_end",
        "id": "c1",
        "name": "read",
        "arguments": {"path": "x"},
    }


def test_event_to_wire_unicode_preserved() -> None:
    """ensure_ascii=False so unicode survives without \\u escapes."""
    from pym.client import TextDelta
    from pym.messages import AssistantMessage

    partial = AssistantMessage()
    wire = event_to_wire(TextDelta(content_index=0, delta="héllo 🚀", partial=partial))
    assert wire is not None
    line = json.dumps(wire, ensure_ascii=False)
    assert "héllo" in line
    assert "🚀" in line
