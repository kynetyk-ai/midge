from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from midge.agent import (
    INTERRUPTED_MESSAGE,
    MAX_TOOL_RESULT_CHARS,
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
    repair_history,
)
from midge.providers import Delta, ToolCallFragment
from midge.providers.openai_compat import encode_messages
from midge.tools import ToolRegistry, tool
from tests.fakes import finish, install, install_provider, say, tcall

_MARKER = object()


async def _edge() -> AsyncIterator[Any]:
    yield _MARKER


async def _collect(agent: Agent, user_input: str) -> list[AgentEvent]:
    return [ev async for ev in agent.stream(user_input)]


async def test_single_turn_text_only() -> None:
    client = Client()
    install(
        client,
        [
            [say("hello"), finish()],
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
    install(
        client,
        [
            # turn 1: tool call
            [
                tcall(index=0, id="c1", name="echo", args='{"text":"hi"}'),
                finish("tool_use"),
            ],
            # turn 2: final text
            [say("done"), finish()],
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
    install(
        client,
        [
            [
                tcall(index=0, id="c1", name="echo", args='{"text":"a"}'),
                tcall(index=1, id="c2", name="echo", args='{"text":"b"}'),
                finish("tool_use"),
            ],
            [say("ok"), finish()],
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
    install(
        client,
        [
            [
                tcall(index=0, id="c1", name="explode", args='{"x":1}'),
                finish("tool_use"),
            ],
            [say("acknowledged"), finish()],
        ],
    )
    agent = Agent(client=client, model="gpt-4o", tools=ToolRegistry([explode]))

    await agent.run("boom")

    tool_result = agent.history[2]
    assert isinstance(tool_result, ToolResultMessage)
    assert tool_result.is_error is True
    assert isinstance(tool_result.content[0], TextContent)
    assert "kaboom" in tool_result.content[0].text


async def _result_text(fn_tool: Any, args: str) -> str:
    client = Client()
    install(
        client,
        [
            [tcall(index=0, id="c1", name=fn_tool.name, args=args), finish("tool_use")],
            [say("ok"), finish()],
        ],
    )
    agent = Agent(client=client, model="m", tools=ToolRegistry([fn_tool]))
    await agent.run("go")
    result = agent.history[2]
    assert isinstance(result, ToolResultMessage) and result.is_error
    assert isinstance(result.content[0], TextContent)
    return result.content[0].text


async def test_a_keyerror_inside_a_tool_is_the_tools_own_error() -> None:
    # #104: a missing note used to be reported as a missing *tool*, and the
    # model stopped calling it.
    @tool
    async def read_note(title: str) -> str:
        raise KeyError(f"No note titled {title!r}")

    text = await _result_text(read_note, '{"title": "nope"}')
    assert "not found" not in text
    assert "No note titled 'nope'" in text


async def test_a_validationerror_inside_a_tool_is_not_blamed_on_the_arguments() -> None:
    from pydantic import BaseModel

    class Row(BaseModel):
        n: int

    @tool
    async def parse(raw: str) -> str:
        return str(Row.model_validate({"n": raw}))

    text = await _result_text(parse, '{"raw": "x"}')
    assert not text.startswith("Invalid arguments")
    assert text.startswith("Tool error: ValidationError")


async def test_bad_arguments_are_still_reported_as_bad_arguments() -> None:
    @tool
    async def add(a: int) -> int:
        return a

    text = await _result_text(add, '{"a": "x"}')
    assert text.startswith("Invalid arguments")


async def test_unknown_tool_becomes_error_result() -> None:
    client = Client()
    install(
        client,
        [
            [
                tcall(index=0, id="c1", name="ghost", args="{}"),
                finish("tool_use"),
            ],
            [say("ok"), finish()],
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
    install(
        client,
        [
            [
                tcall(index=0, id="c1", name="add", args='{"a":"oops"}'),
                finish("tool_use"),
            ],
            [say("ok"), finish()],
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

    client = Client()
    # A RuntimeError is not retryable, so exactly one request is made and the
    # loop must stop rather than take another turn on an errored message.
    provider = install_provider(client, [[RuntimeError("upstream blew up")]])
    agent = Agent(client=client, model="gpt-4o", tools=ToolRegistry([echo]))

    msg = await agent.run("trigger error")

    assert msg.stop_reason == "error"
    assert provider.attempts == 1


async def test_system_prompt_passed_to_client() -> None:
    client = Client()
    captured = install(
        client,
        [[say("hi"), finish()]],
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
    install(
        client,
        [
            [say("first"), finish()],
            [say("second"), finish()],
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
    install(
        client,
        [
            [
                tcall(index=0, id="c1", name="echo", args='{"text":"x"}'),
                finish("tool_use"),
            ],
            [say("ok"), finish()],
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
    install(
        client,
        [[say("hi"), finish()]],
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
    captured = install(
        client,
        [[say("hi"), finish()]],
    )
    agent = Agent(client=client, model="gpt-4o")
    await agent.run("ping")
    assert len(captured) == 1
    # No `tools` kwarg should be passed when registry is empty
    assert "tools" not in captured[0] or not captured[0].get("tools")


async def test_user_message_can_be_passed_directly() -> None:
    client = Client()
    install(
        client,
        [[say("hi"), finish()]],
    )
    agent = Agent(client=client, model="gpt-4o")

    user = UserMessage(content="hello there")
    await agent.run(user)

    assert agent.history[0] is user


async def test_cancel_during_tool_execution_closes_out_tool_calls() -> None:
    """An unanswered tool call makes every later turn unusable — see issue #27."""
    started = asyncio.Event()

    @tool
    async def slow(x: str) -> str:
        started.set()
        await asyncio.sleep(30)
        return "done"

    registry = ToolRegistry()
    registry.add(slow)

    client = Client()
    install(
        client,
        [
            [
                tcall(index=0, id="c1", name="slow", args='{"x":"1"}'),
                finish("tool_use"),
            ]
        ],
    )
    agent = Agent(client=client, model="gpt-4o", tools=registry)

    async def run() -> None:
        async for _ in agent.stream("go"):
            pass

    task = asyncio.create_task(run())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    result = agent.history[-1]
    assert isinstance(result, ToolResultMessage)
    assert result.tool_call_id == "c1"
    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == INTERRUPTED_MESSAGE

    # The next request must carry a `tool` message for every `tool_call` id.
    captured = install(
        client, [[say("hi"), finish()]]
    )
    await agent.run("next")
    sent = encode_messages(repair_history(agent.history[:-1]))
    answered = {m["tool_call_id"] for m in sent if m.get("role") == "tool"}
    requested = {
        tc["id"]
        for m in sent
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    assert requested and requested == answered
    assert captured


async def test_cancel_only_closes_out_unfinished_tool_calls() -> None:
    started = asyncio.Event()

    @tool
    async def fast(x: str) -> str:
        return "quick"

    @tool
    async def slow(x: str) -> str:
        started.set()
        await asyncio.sleep(30)
        return "done"

    registry = ToolRegistry()
    registry.add(fast)
    registry.add(slow)

    client = Client()
    install(
        client,
        [
            [
                # Both calls in one chunk, which is what a provider does when it
                # emits parallel calls together rather than one per chunk.
                Delta(
                    tool_calls=(
                        ToolCallFragment(index=0, id="c1", name="fast", arguments='{"x":"1"}'),
                        ToolCallFragment(index=1, id="c2", name="slow", arguments='{"x":"2"}'),
                    )
                ),
                finish("tool_use"),
            ]
        ],
    )
    agent = Agent(client=client, model="gpt-4o", tools=registry)

    async def run() -> None:
        async for _ in agent.stream("go"):
            pass

    task = asyncio.create_task(run())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    results = {
        m.tool_call_id: m for m in agent.history if isinstance(m, ToolResultMessage)
    }
    assert set(results) == {"c1", "c2"}
    assert not results["c1"].is_error
    assert results["c2"].is_error


async def test_reentrant_stream_is_rejected() -> None:
    """Two concurrent turns would interleave appends into history — issue #33."""
    client = Client()
    started = asyncio.Event()

    @tool
    async def slow(x: str) -> str:
        started.set()
        await asyncio.sleep(30)
        return "done"

    registry = ToolRegistry()
    registry.add(slow)
    install(
        client,
        [
            [
                tcall(index=0, id="c1", name="slow", args='{"x":"1"}'),
                finish("tool_use"),
            ]
        ],
    )
    agent = Agent(client=client, model="gpt-4o", tools=registry)

    async def run() -> None:
        async for _ in agent.stream("first"):
            pass

    task = asyncio.create_task(run())
    await started.wait()

    with pytest.raises(RuntimeError, match="already running"):
        async for _ in agent.stream("second"):
            pass

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # And the guard releases, so the next turn works.
    install(client, [[say("ok"), finish()]])
    await agent.run("third")


async def test_truncated_message_fails_tool_calls_unexecuted() -> None:
    ran = False

    @tool
    async def touch(path: str = "/tmp/x") -> str:
        nonlocal ran
        ran = True
        return "ran"

    registry = ToolRegistry()
    registry.add(touch)
    client = Client()
    install(
        client,
        [
            [
                tcall(index=0, id="t1", name="touch", args='{"pa'),
                finish("length"),
            ]
        ],
    )
    agent = Agent(client=client, model="gpt-4o", tools=registry)
    await _collect(agent, "go")

    assert ran is False
    result = agent.history[-1]
    assert isinstance(result, ToolResultMessage)
    assert result.is_error
    assert result.tool_call_id == "t1"


async def test_unparseable_tool_arguments_are_not_executed() -> None:
    ran = False

    @tool
    async def maybe(path: str = "default") -> str:
        nonlocal ran
        ran = True
        return "ran"

    registry = ToolRegistry()
    registry.add(maybe)
    client = Client()
    install(
        client,
        [
            [
                tcall(index=0, id="m1", name="maybe", args='{"path": "trunc'),
                finish("tool_use"),
            ],
            [say("ok"), finish()],
        ],
    )
    agent = Agent(client=client, model="gpt-4o", tools=registry)
    await _collect(agent, "go")

    # A tool whose params all have defaults would otherwise silently run.
    assert ran is False
    results = [m for m in agent.history if isinstance(m, ToolResultMessage)]
    assert len(results) == 1
    assert results[0].is_error
    assert isinstance(results[0].content[0], TextContent)
    assert "could not be parsed" in results[0].content[0].text


async def test_oversized_tool_result_is_truncated() -> None:
    @tool
    async def big(n: int = 0) -> str:
        return "x" * (MAX_TOOL_RESULT_CHARS + 5_000)

    registry = ToolRegistry()
    registry.add(big)
    client = Client()
    install(
        client,
        [
            [
                tcall(index=0, id="b1", name="big", args="{}"),
                finish("tool_use"),
            ],
            [say("ok"), finish()],
        ],
    )
    agent = Agent(client=client, model="gpt-4o", tools=registry)
    await _collect(agent, "go")

    result = next(m for m in agent.history if isinstance(m, ToolResultMessage))
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    assert len(text) < MAX_TOOL_RESULT_CHARS + 200
    assert "truncated" in text


async def test_generator_exit_closes_out_tool_calls() -> None:
    """Breaking out of the stream must not orphan an in-flight tool call."""
    started = asyncio.Event()

    @tool
    async def slow(x: str) -> str:
        started.set()
        await asyncio.sleep(30)
        return "done"

    registry = ToolRegistry()
    registry.add(slow)
    client = Client()
    install(
        client,
        [
            [
                tcall(index=0, id="g1", name="slow", args='{"x":"1"}'),
                finish("tool_use"),
            ]
        ],
    )
    agent = Agent(client=client, model="gpt-4o", tools=registry)

    async def consume() -> None:
        async for ev in agent.stream("go"):
            if isinstance(ev, ToolExecutionStart):
                break

    task = asyncio.create_task(consume())
    await task
    # Force finalization of the abandoned generator.
    for _ in range(3):
        await asyncio.sleep(0)
    import gc

    gc.collect()
    for _ in range(3):
        await asyncio.sleep(0)

    wire = encode_messages(repair_history(agent.history))
    requested = {
        tc["id"]
        for m in wire
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    answered = {m["tool_call_id"] for m in wire if m.get("role") == "tool"}
    assert requested == answered


# --- #101: what a message with several tool calls promises -----------------


def _calls(*specs: tuple[str, str]) -> list[Any]:
    return [
        tcall(index=i, id=f"c{i}", name=name, args=args) for i, (name, args) in enumerate(specs)
    ]


async def test_a_mutating_call_waits_for_the_one_before_it() -> None:
    # The phase-1 race: copy a file, then edit the copy, in one message.
    log: list[str] = []

    @tool
    async def write_it(path: str) -> str:
        log.append(f"start write {path}")
        await asyncio.sleep(0.05)
        log.append(f"end write {path}")
        return "written"

    @tool
    async def edit_it(path: str) -> str:
        log.append(f"start edit {path}")
        return "edited"

    client = Client()
    install(
        client,
        [
            [*_calls(("write_it", '{"path":"a"}'), ("edit_it", '{"path":"a"}')),
             finish("tool_use")],
            [say("ok"), finish()],
        ],
    )
    agent = Agent(client=client, model="m", tools=ToolRegistry([write_it, edit_it]))
    await agent.run("go")

    assert log == ["start write a", "end write a", "start edit a"]


async def test_read_only_calls_run_together() -> None:
    started: list[str] = []
    both = asyncio.Event()

    @tool(read_only=True)
    async def peek(name: str) -> str:
        started.append(name)
        if len(started) == 2:
            both.set()
        # Serialized, the first would wait here forever for the second.
        await asyncio.wait_for(both.wait(), timeout=1)
        return name

    client = Client()
    install(
        client,
        [
            [*_calls(("peek", '{"name":"a"}'), ("peek", '{"name":"b"}')), finish("tool_use")],
            [say("ok"), finish()],
        ],
    )
    agent = Agent(client=client, model="m", tools=ToolRegistry([peek]))
    await agent.run("go")

    results = [m for m in agent.history if isinstance(m, ToolResultMessage)]
    assert [r.is_error for r in results] == [False, False]


async def test_reads_around_a_write_are_kept_on_their_side_of_it() -> None:
    log: list[str] = []

    @tool(read_only=True)
    async def peek(name: str) -> str:
        log.append(f"start {name}")
        await asyncio.sleep(0.02)
        log.append(f"end {name}")
        return name

    @tool
    async def poke(name: str) -> str:
        log.append(f"start {name}")
        await asyncio.sleep(0.02)
        log.append(f"end {name}")
        return name

    client = Client()
    install(
        client,
        [
            [*_calls(("peek", '{"name":"r1"}'), ("poke", '{"name":"w"}'),
                     ("peek", '{"name":"r2"}')), finish("tool_use")],
            [say("ok"), finish()],
        ],
    )
    agent = Agent(client=client, model="m", tools=ToolRegistry([peek, poke]))
    await agent.run("go")

    assert log == ["start r1", "end r1", "start w", "end w", "start r2", "end r2"]
    # Results still come back in the order the calls were made.
    ids = [m.tool_call_id for m in agent.history if isinstance(m, ToolResultMessage)]
    assert ids == ["c0", "c1", "c2"]


async def test_a_cancel_mid_sequence_answers_every_call() -> None:
    started = asyncio.Event()

    @tool
    async def hang(x: str) -> str:
        started.set()
        await asyncio.Event().wait()
        return x

    @tool
    async def after(x: str) -> str:
        return x

    client = Client()
    install(
        client,
        [[*_calls(("hang", '{"x":"1"}'), ("after", '{"x":"2"}')), finish("tool_use")]],
    )
    agent = Agent(client=client, model="m", tools=ToolRegistry([hang, after]))

    async def run() -> None:
        async for _ in agent.stream("go"):
            pass

    task = asyncio.create_task(run())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    results = [m for m in agent.history if isinstance(m, ToolResultMessage)]
    assert [r.tool_call_id for r in results] == ["c0", "c1"]
    assert all(r.is_error for r in results)
    texts = [r.content[0].text for r in results if isinstance(r.content[0], TextContent)]
    assert texts == [INTERRUPTED_MESSAGE, INTERRUPTED_MESSAGE]


async def test_between_requests_runs_at_each_loop_edge() -> None:
    @tool
    async def echo(text: str) -> str:
        return f"echoed:{text}"

    client = Client()
    install(
        client,
        [
            # turn 1: tool call
            [
                tcall(index=0, id="c1", name="echo", args='{"text":"hi"}'),
                finish("tool_use"),
            ],
            # turn 2: final text
            [say("done"), finish()],
        ],
    )
    agent = Agent(client=client, model="gpt-4o", tools=ToolRegistry([echo]))
    agent.between_requests = cast(Any, _edge)

    events: list[AgentEvent] = [ev async for ev in agent.stream("please echo hi")]
    raw: list[Any] = events

    markers = [ev for ev in raw if ev is _MARKER]
    assert len(markers) == 1
    assert isinstance(events[-1], AgentEnd)
    assert raw.index(_MARKER) < raw.index(events[-1])
