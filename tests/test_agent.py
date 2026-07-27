from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

from midge.agent import (
    Agent,
    AgentEnd,
    AgentEvent,
    ToolExecutionEnd,
    ToolExecutionStart,
)
from midge.client import Client
from midge.messages import (
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from midge.tools import ToolRegistry, tool


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


def _install_turns(
    client: Client, turns: list[list[Any]]
) -> list[dict[str, Any]]:
    """Pre-canned per-turn chunk sequences. Returns a captured list of `create()` kwargs."""
    captured: list[dict[str, Any]] = []
    iterator = iter(turns)

    async def create(**kwargs: Any) -> _FakeStream:
        captured.append(kwargs)
        return _FakeStream(next(iterator))

    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return captured


async def _collect(agent: Agent, user_input: str) -> list[AgentEvent]:
    return [ev async for ev in agent.stream(user_input)]


async def test_single_turn_text_only() -> None:
    client = Client()
    _install_turns(
        client,
        [
            [_chunk(content="hello"), _chunk(finish_reason="stop")],
        ],
    )
    agent = Agent(client=client, model="gpt-4o")

    msg = await agent.run("hi")

    assert isinstance(msg, AssistantMessage)
    assert msg.stop_reason == "stop"
    assert isinstance(msg.content[0], TextContent)
    assert msg.content[0].text == "hello"
    assert len(agent.history) == 2
    assert isinstance(agent.history[0], UserMessage)
    assert isinstance(agent.history[1], AssistantMessage)


async def test_single_tool_call_then_finish() -> None:
    @tool
    async def echo(text: str) -> str:
        return f"echoed:{text}"

    client = Client()
    _install_turns(
        client,
        [
            # turn 1: tool call
            [
                _chunk(
                    tool_calls=[_tcd(index=0, id="c1", name="echo", arguments='{"text":"hi"}')]
                ),
                _chunk(finish_reason="tool_calls"),
            ],
            # turn 2: final text
            [_chunk(content="done"), _chunk(finish_reason="stop")],
        ],
    )
    agent = Agent(client=client, model="gpt-4o", tools=ToolRegistry([echo]))

    msg = await agent.run("please echo hi")

    assert msg.stop_reason == "stop"
    assert isinstance(msg.content[0], TextContent)
    assert msg.content[0].text == "done"

    # history: user, assistant_with_tool_call, tool_result, assistant_text
    assert len(agent.history) == 4
    tool_result = agent.history[2]
    assert isinstance(tool_result, ToolResultMessage)
    assert tool_result.tool_call_id == "c1"
    assert tool_result.is_error is False
    assert isinstance(tool_result.content[0], TextContent)
    assert tool_result.content[0].text == "echoed:hi"


async def test_parallel_tool_calls() -> None:
    @tool
    async def echo(text: str) -> str:
        return text.upper()

    client = Client()
    _install_turns(
        client,
        [
            [
                _chunk(
                    tool_calls=[_tcd(index=0, id="c1", name="echo", arguments='{"text":"a"}')]
                ),
                _chunk(
                    tool_calls=[_tcd(index=1, id="c2", name="echo", arguments='{"text":"b"}')]
                ),
                _chunk(finish_reason="tool_calls"),
            ],
            [_chunk(content="ok"), _chunk(finish_reason="stop")],
        ],
    )
    agent = Agent(client=client, model="gpt-4o", tools=ToolRegistry([echo]))

    events = await _collect(agent, "do both")

    starts = [e for e in events if isinstance(e, ToolExecutionStart)]
    ends = [e for e in events if isinstance(e, ToolExecutionEnd)]
    assert [s.tool_call.id for s in starts] == ["c1", "c2"]
    assert [e.tool_call.id for e in ends] == ["c1", "c2"]
    assert isinstance(ends[0].result.content[0], TextContent)
    assert ends[0].result.content[0].text == "A"
    assert isinstance(ends[1].result.content[0], TextContent)
    assert ends[1].result.content[0].text == "B"


async def test_tool_raising_exception_becomes_error_result() -> None:
    @tool
    async def explode(x: int) -> int:
        raise RuntimeError("kaboom")

    client = Client()
    _install_turns(
        client,
        [
            [
                _chunk(
                    tool_calls=[
                        _tcd(index=0, id="c1", name="explode", arguments='{"x":1}')
                    ]
                ),
                _chunk(finish_reason="tool_calls"),
            ],
            [_chunk(content="acknowledged"), _chunk(finish_reason="stop")],
        ],
    )
    agent = Agent(client=client, model="gpt-4o", tools=ToolRegistry([explode]))

    await agent.run("boom")

    tool_result = agent.history[2]
    assert isinstance(tool_result, ToolResultMessage)
    assert tool_result.is_error is True
    assert isinstance(tool_result.content[0], TextContent)
    assert "kaboom" in tool_result.content[0].text


async def test_unknown_tool_becomes_error_result() -> None:
    client = Client()
    _install_turns(
        client,
        [
            [
                _chunk(
                    tool_calls=[
                        _tcd(index=0, id="c1", name="ghost", arguments="{}")
                    ]
                ),
                _chunk(finish_reason="tool_calls"),
            ],
            [_chunk(content="ok"), _chunk(finish_reason="stop")],
        ],
    )
    agent = Agent(client=client, model="gpt-4o")  # no tools registered

    await agent.run("call ghost")

    tool_result = agent.history[2]
    assert isinstance(tool_result, ToolResultMessage)
    assert tool_result.is_error is True
    assert isinstance(tool_result.content[0], TextContent)
    assert "ghost" in tool_result.content[0].text


async def test_invalid_tool_arguments_become_error_result() -> None:
    @tool
    async def add(a: int, b: int) -> int:
        return a + b

    client = Client()
    _install_turns(
        client,
        [
            [
                _chunk(
                    tool_calls=[
                        _tcd(index=0, id="c1", name="add", arguments='{"a":"oops"}')
                    ]
                ),
                _chunk(finish_reason="tool_calls"),
            ],
            [_chunk(content="ok"), _chunk(finish_reason="stop")],
        ],
    )
    agent = Agent(client=client, model="gpt-4o", tools=ToolRegistry([add]))

    await agent.run("bad call")

    tool_result = agent.history[2]
    assert isinstance(tool_result, ToolResultMessage)
    assert tool_result.is_error is True


async def test_error_stop_reason_terminates_loop() -> None:
    @tool
    async def echo(text: str) -> str:
        return text

    captured_calls = 0

    class FailFirstStream:
        def __init__(self) -> None:
            self._raised = False

        def __aiter__(self) -> FailFirstStream:
            return self

        async def __anext__(self) -> Any:
            if not self._raised:
                self._raised = True
                raise RuntimeError("upstream blew up")
            raise StopAsyncIteration

    async def create(**kwargs: Any) -> FailFirstStream:
        nonlocal captured_calls
        captured_calls += 1
        return FailFirstStream()

    client = Client()
    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    agent = Agent(client=client, model="gpt-4o", tools=ToolRegistry([echo]))

    msg = await agent.run("trigger error")

    assert msg.stop_reason == "error"
    assert captured_calls == 1  # loop did not retry


async def test_system_prompt_passed_to_client() -> None:
    client = Client()
    captured = _install_turns(
        client,
        [[_chunk(content="hi"), _chunk(finish_reason="stop")]],
    )
    agent = Agent(
        client=client,
        model="gpt-4o",
        system_prompt="You are helpful.",
    )

    await agent.run("ping")

    assert len(captured) == 1
    msgs = captured[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "You are helpful."}
    assert msgs[1]["role"] == "user"


async def test_multi_turn_history_grows() -> None:
    client = Client()
    _install_turns(
        client,
        [
            [_chunk(content="first"), _chunk(finish_reason="stop")],
            [_chunk(content="second"), _chunk(finish_reason="stop")],
        ],
    )
    agent = Agent(client=client, model="gpt-4o")

    await agent.run("a")
    assert len(agent.history) == 2

    await agent.run("b")
    assert len(agent.history) == 4
    assert isinstance(agent.history[2], UserMessage)
    assert agent.history[2].content == "b"


async def test_agent_end_event_has_only_this_turns_messages() -> None:
    @tool
    async def echo(text: str) -> str:
        return text

    client = Client()
    _install_turns(
        client,
        [
            [
                _chunk(
                    tool_calls=[_tcd(index=0, id="c1", name="echo", arguments='{"text":"x"}')]
                ),
                _chunk(finish_reason="tool_calls"),
            ],
            [_chunk(content="ok"), _chunk(finish_reason="stop")],
        ],
    )
    agent = Agent(client=client, model="gpt-4o", tools=ToolRegistry([echo]))

    events = await _collect(agent, "go")
    end = events[-1]
    assert isinstance(end, AgentEnd)
    # this turn produced: user, assistant(tool_call), tool_result, assistant(text)
    assert len(end.new_messages) == 4
    assert isinstance(end.new_messages[0], UserMessage)
    assert isinstance(end.new_messages[-1], AssistantMessage)


async def test_event_sequence_for_simple_run() -> None:
    client = Client()
    _install_turns(
        client,
        [[_chunk(content="hi"), _chunk(finish_reason="stop")]],
    )
    agent = Agent(client=client, model="gpt-4o")

    events = await _collect(agent, "ping")
    types = [type(ev).__name__ for ev in events]

    # last must be AgentEnd; Done must appear before it; no tool events.
    assert types[-1] == "AgentEnd"
    assert "Done" in types
    assert "ToolExecutionStart" not in types
    assert "ToolExecutionEnd" not in types


async def test_run_no_tools_calls_client_once() -> None:
    client = Client()
    captured = _install_turns(
        client,
        [[_chunk(content="hi"), _chunk(finish_reason="stop")]],
    )
    agent = Agent(client=client, model="gpt-4o")
    await agent.run("ping")
    assert len(captured) == 1
    # No `tools` kwarg should be passed when registry is empty
    assert "tools" not in captured[0] or not captured[0].get("tools")


async def test_user_message_can_be_passed_directly() -> None:
    client = Client()
    _install_turns(
        client,
        [[_chunk(content="hi"), _chunk(finish_reason="stop")]],
    )
    agent = Agent(client=client, model="gpt-4o")

    user = UserMessage(content="hello there")
    await agent.run(user)

    assert agent.history[0] is user
