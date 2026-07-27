from __future__ import annotations

import logging
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import pytest

from midge.agent import Agent, AgentEvent, ToolExecutionEnd, ToolExecutionStart
from midge.client import Client
from midge.compaction import compact
from midge.hooks import (
    BeforeCompact,
    BeforeProviderRequest,
    CancelResult,
    CompactResult,
    Context,
    ContextResult,
    Hooks,
    MessageEnd,
    ProviderRequestResult,
    ProviderResponseResult,
    SessionStart,
    ToolCallEvent,
    ToolCallResult,
    ToolResultEvent,
    ToolResultResult,
    TurnStart,
    TurnStartResult,
)
from midge.messages import AssistantMessage, TextContent, UserMessage
from midge.tools import tool

# --- shared fakes (mirrors tests/test_agent.py) ---------------------------


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


def _install_turns(client: Client, turns: list[list[Any]]) -> list[dict[str, Any]]:
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


def _tool_turn(*calls: tuple[str, str, str]) -> list[Any]:
    """Build a turn that emits the given (id, name, json_args) tool calls."""
    chunks = [
        _chunk(tool_calls=[_tcd(index=i, id=cid, name=name, arguments=args)])
        for i, (cid, name, args) in enumerate(calls)
    ]
    chunks.append(_chunk(finish_reason="tool_calls"))
    return chunks


# --- registry mechanics ---------------------------------------------------


async def test_observe_sees_every_event_and_return_is_ignored() -> None:
    hooks = Hooks()
    seen: list[str] = []
    hooks.observe(lambda ev, ctx: seen.append(ev.type) or ContextResult(messages=[]))

    await hooks.emit(MessageEnd(message=AssistantMessage()))
    await hooks.emit(Context(messages=[]))

    assert seen == ["message_end", "context"]


async def test_unsubscribe_actually_unsubscribes() -> None:
    hooks = Hooks()
    calls: list[str] = []
    off_obs = hooks.observe(lambda ev, ctx: calls.append("obs"))
    off_on = hooks.on("message_end", lambda ev, ctx: calls.append("on"))

    await hooks.emit(MessageEnd(message=AssistantMessage()))
    off_obs()
    off_on()
    await hooks.emit(MessageEnd(message=AssistantMessage()))

    assert calls == ["obs", "on"]


async def test_sync_and_async_handlers_both_work() -> None:
    hooks = Hooks()

    async def async_handler(ev: Any, ctx: Any) -> ContextResult:
        return ContextResult(messages=[UserMessage(content="async")])

    hooks.on("context", async_handler)
    res = await hooks.emit(Context(messages=[]))
    assert isinstance(res, ContextResult)
    assert res.messages is not None
    assert len(res.messages) == 1

    hooks2 = Hooks()
    hooks2.on("context", lambda ev, ctx: ContextResult(messages=[UserMessage(content="sync")]))
    res2 = await hooks2.emit(Context(messages=[]))
    assert isinstance(res2, ContextResult)


async def test_context_object_is_passed_to_handlers() -> None:
    sentinel = object()
    hooks = Hooks(sentinel)
    got: list[Any] = []
    hooks.on("message_end", lambda ev, ctx: got.append(ctx))
    await hooks.emit(MessageEnd(message=AssistantMessage()))
    assert got == [sentinel]


async def test_no_handlers_returns_none() -> None:
    hooks = Hooks()
    assert await hooks.emit(Context(messages=[])) is None


async def test_cleanups_run_on_clear() -> None:
    hooks = Hooks()
    ran: list[str] = []
    hooks.add_cleanup(lambda: ran.append("cleanup"))
    hooks.on("message_end", lambda ev, ctx: None)

    await hooks.clear()

    assert ran == ["cleanup"]
    assert await hooks.emit(MessageEnd(message=AssistantMessage())) is None


# --- error policy ---------------------------------------------------------


async def test_raising_handler_is_logged_and_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hooks = Hooks()

    def boom(ev: Any, ctx: Any) -> None:
        raise RuntimeError("nope")

    hooks.on("context", boom, source="/ext/bad.py")
    hooks.on("context", lambda ev, ctx: ContextResult(messages=[UserMessage(content="ok")]))

    with caplog.at_level(logging.WARNING, logger="midge.hooks"):
        res = await hooks.emit(Context(messages=[]))

    assert isinstance(res, ContextResult)
    assert any("/ext/bad.py" in rec.message for rec in caplog.records)


async def test_error_mode_raise_propagates() -> None:
    hooks = Hooks(error_mode="raise")

    def boom(ev: Any, ctx: Any) -> None:
        raise RuntimeError("nope")

    hooks.on("context", boom)
    with pytest.raises(RuntimeError, match="nope"):
        await hooks.emit(Context(messages=[]))


# --- reduction rules in isolation -----------------------------------------


async def test_context_chains_each_handler_sees_previous_output() -> None:
    hooks = Hooks()

    def first(ev: Context, ctx: Any) -> ContextResult:
        return ContextResult(messages=[*ev.messages, UserMessage(content="a")])

    def second(ev: Context, ctx: Any) -> ContextResult:
        assert len(ev.messages) == 1  # sees first's output
        return ContextResult(messages=[*ev.messages, UserMessage(content="b")])

    hooks.on("context", first)
    hooks.on("context", second)

    res = await hooks.emit(Context(messages=[]))
    assert isinstance(res, ContextResult)
    assert res.messages is not None
    assert [m.content for m in res.messages] == ["a", "b"]


async def test_first_cancel_or_last_short_circuits() -> None:
    hooks = Hooks()
    seen: list[str] = []
    hooks.on("session_start", lambda ev, ctx: seen.append("one") or CancelResult(cancel=True))
    hooks.on("session_start", lambda ev, ctx: seen.append("two"))

    res = await hooks.emit(SessionStart())

    assert isinstance(res, CancelResult)
    assert res.cancel is True
    assert seen == ["one"]


async def test_tool_call_early_exits_on_block() -> None:
    hooks = Hooks()
    seen: list[str] = []
    hooks.on("tool_call", lambda ev, ctx: seen.append("one") or ToolCallResult(block=True))
    hooks.on("tool_call", lambda ev, ctx: seen.append("two"))

    tc = _make_tool_call()
    res = await hooks.emit(ToolCallEvent(tool_call=tc))

    assert isinstance(res, ToolCallResult)
    assert res.block is True
    assert seen == ["one"]


async def test_tool_call_argument_rewrite_visible_to_later_handlers() -> None:
    hooks = Hooks()
    observed: list[dict[str, Any]] = []

    hooks.on("tool_call", lambda ev, ctx: ToolCallResult(arguments={"text": "rewritten"}))
    hooks.on("tool_call", lambda ev, ctx: observed.append(ev.tool_call.arguments))

    res = await hooks.emit(ToolCallEvent(tool_call=_make_tool_call()))

    assert observed == [{"text": "rewritten"}]
    assert isinstance(res, ToolCallResult)
    assert res.arguments == {"text": "rewritten"}


async def test_tool_result_patches_accumulate() -> None:
    hooks = Hooks()
    hooks.on("tool_result", lambda ev, ctx: ToolResultResult(content=[TextContent(text="x")]))
    hooks.on("tool_result", lambda ev, ctx: ToolResultResult(is_error=True))

    res = await hooks.emit(ToolResultEvent(tool_call=_make_tool_call()))

    assert isinstance(res, ToolResultResult)
    assert res.is_error is True
    assert res.content is not None
    assert isinstance(res.content[0], TextContent)
    assert res.content[0].text == "x"


async def test_turn_start_collects_messages_and_chains_prompt() -> None:
    hooks = Hooks()
    hooks.on(
        "turn_start",
        lambda ev, ctx: TurnStartResult(
            messages=[UserMessage(content="memory")], system_prompt="first"
        ),
    )
    hooks.on(
        "turn_start",
        lambda ev, ctx: TurnStartResult(system_prompt=f"{ev.system_prompt}+second"),
    )

    res = await hooks.emit(TurnStart(user_message=UserMessage(content="hi"), system_prompt="base"))

    assert isinstance(res, TurnStartResult)
    assert res.system_prompt == "first+second"
    assert res.messages is not None
    assert len(res.messages) == 1


async def test_provider_request_transform_chains() -> None:
    hooks = Hooks()
    hooks.on("before_provider_request", lambda ev, ctx: ProviderRequestResult(model="m2"))
    hooks.on(
        "before_provider_request",
        lambda ev, ctx: ProviderRequestResult(kwargs={"temperature": 0.1}),
    )

    res = await hooks.emit(BeforeProviderRequest(model="m1", system="s"))

    assert isinstance(res, ProviderRequestResult)
    assert res.model == "m2"
    assert res.system == "s"
    assert res.kwargs == {"temperature": 0.1}


async def test_unchanged_transform_returns_none() -> None:
    hooks = Hooks()
    hooks.on("context", lambda ev, ctx: None)
    assert await hooks.emit(Context(messages=[])) is None


# --- Agent integration ----------------------------------------------------


def _make_tool_call() -> Any:
    from midge.messages import ToolCall

    return ToolCall(id="c1", name="echo", arguments={"text": "hi"})


@tool
async def echo(text: str) -> str:
    return f"echoed:{text}"


@tool
async def danger(text: str) -> str:
    return "SHOULD NOT RUN"


def _registry() -> Any:
    from midge.tools import ToolRegistry

    reg = ToolRegistry()
    reg.add(echo)
    reg.add(danger)
    return reg


async def test_agent_without_hooks_is_unchanged() -> None:
    client = Client()
    _install_turns(client, [[_chunk(content="hello"), _chunk(finish_reason="stop")]])
    agent = Agent(client=client, model="gpt-4o")

    msg = await agent.run("hi")

    assert msg.stop_reason == "stop"
    assert len(agent.history) == 2


async def test_blocked_tool_never_executes_and_yields_error_result() -> None:
    client = Client()
    _install_turns(
        client,
        [
            _tool_turn(("c1", "danger", '{"text": "x"}')),
            [_chunk(content="done"), _chunk(finish_reason="stop")],
        ],
    )
    hooks = Hooks()
    hooks.on(
        "tool_call",
        lambda ev, ctx: ToolCallResult(block=True, reason="denied by policy")
        if ev.tool_call.name == "danger"
        else None,
    )
    agent = Agent(client=client, model="gpt-4o", tools=_registry(), hooks=hooks)

    events = await _collect(agent, "go")

    ends = [e for e in events if isinstance(e, ToolExecutionEnd)]
    starts = [e for e in events if isinstance(e, ToolExecutionStart)]
    assert len(starts) == 1  # blocked calls still surface to the UI
    assert len(ends) == 1
    assert ends[0].result.is_error is True
    assert isinstance(ends[0].result.content[0], TextContent)
    assert ends[0].result.content[0].text == "denied by policy"
    assert "SHOULD NOT RUN" not in str(ends[0].result.content)


async def test_ordering_preserved_when_some_calls_blocked() -> None:
    """The `zip(strict=True)` invariant: results must line up with tool_calls
    even when the blocked ones are removed from the gather."""
    client = Client()
    _install_turns(
        client,
        [
            _tool_turn(
                ("c1", "echo", '{"text": "one"}'),
                ("c2", "danger", '{"text": "two"}'),
                ("c3", "echo", '{"text": "three"}'),
            ),
            [_chunk(content="done"), _chunk(finish_reason="stop")],
        ],
    )
    hooks = Hooks()
    hooks.on(
        "tool_call",
        lambda ev, ctx: ToolCallResult(block=True, reason="no")
        if ev.tool_call.name == "danger"
        else None,
    )
    agent = Agent(client=client, model="gpt-4o", tools=_registry(), hooks=hooks)

    events = await _collect(agent, "go")
    ends = [e for e in events if isinstance(e, ToolExecutionEnd)]

    assert [e.tool_call.id for e in ends] == ["c1", "c2", "c3"]
    assert [e.result.tool_call_id for e in ends] == ["c1", "c2", "c3"]
    assert [e.result.is_error for e in ends] == [False, True, False]
    texts = [e.result.content[0].text for e in ends]  # type: ignore[union-attr]
    assert texts[0] == "echoed:one"
    assert texts[1] == "no"
    assert texts[2] == "echoed:three"


async def test_tool_argument_rewrite_reaches_the_tool() -> None:
    client = Client()
    _install_turns(
        client,
        [
            _tool_turn(("c1", "echo", '{"text": "original"}')),
            [_chunk(content="done"), _chunk(finish_reason="stop")],
        ],
    )
    hooks = Hooks()
    hooks.on("tool_call", lambda ev, ctx: ToolCallResult(arguments={"text": "patched"}))
    agent = Agent(client=client, model="gpt-4o", tools=_registry(), hooks=hooks)

    events = await _collect(agent, "go")
    ends = [e for e in events if isinstance(e, ToolExecutionEnd)]

    assert isinstance(ends[0].result.content[0], TextContent)
    assert ends[0].result.content[0].text == "echoed:patched"


async def test_tool_result_patch_applies_to_history() -> None:
    client = Client()
    _install_turns(
        client,
        [
            _tool_turn(("c1", "echo", '{"text": "secret"}')),
            [_chunk(content="done"), _chunk(finish_reason="stop")],
        ],
    )
    hooks = Hooks()
    hooks.on(
        "tool_result",
        lambda ev, ctx: ToolResultResult(content=[TextContent(text="[redacted]")]),
    )
    agent = Agent(client=client, model="gpt-4o", tools=_registry(), hooks=hooks)

    events = await _collect(agent, "go")
    ends = [e for e in events if isinstance(e, ToolExecutionEnd)]

    assert isinstance(ends[0].result.content[0], TextContent)
    assert ends[0].result.content[0].text == "[redacted]"
    # and the patched version is what lands in history
    assert any(
        getattr(m, "content", None) and "[redacted]" in str(m.content) for m in agent.history
    )


async def test_context_hook_changes_what_provider_sees_without_mutating_history() -> None:
    client = Client()
    captured = _install_turns(client, [[_chunk(content="ok"), _chunk(finish_reason="stop")]])
    hooks = Hooks()
    hooks.on(
        "context",
        lambda ev, ctx: ContextResult(messages=[UserMessage(content="INJECTED")]),
    )
    agent = Agent(client=client, model="gpt-4o", hooks=hooks)

    await agent.run("real question")

    sent = captured[0]["messages"]
    assert any("INJECTED" in str(m) for m in sent)
    # history keeps the true user message
    assert isinstance(agent.history[0], UserMessage)
    assert agent.history[0].content == "real question"


async def test_provider_request_hook_overrides_model() -> None:
    client = Client()
    captured = _install_turns(client, [[_chunk(content="ok"), _chunk(finish_reason="stop")]])
    hooks = Hooks()
    hooks.on("before_provider_request", lambda ev, ctx: ProviderRequestResult(model="override"))
    agent = Agent(client=client, model="gpt-4o", hooks=hooks)

    await agent.run("hi")

    assert captured[0]["model"] == "override"


async def test_after_provider_response_replaces_message_before_history() -> None:
    client = Client()
    _install_turns(client, [[_chunk(content="original"), _chunk(finish_reason="stop")]])
    hooks = Hooks()
    seen_by_message_end: list[str] = []

    hooks.on(
        "after_provider_response",
        lambda ev, ctx: ProviderResponseResult(
            message=AssistantMessage(content=[TextContent(text="replaced")], stop_reason="stop")
        ),
    )
    hooks.on(
        "message_end",
        lambda ev, ctx: seen_by_message_end.append(str(ev.message.content)),
    )
    agent = Agent(client=client, model="gpt-4o", hooks=hooks)

    await agent.run("hi")

    assert "replaced" in str(agent.history[1])
    assert "replaced" in seen_by_message_end[0]


async def test_turn_start_prompt_override_reaches_provider() -> None:
    client = Client()
    captured = _install_turns(client, [[_chunk(content="ok"), _chunk(finish_reason="stop")]])
    hooks = Hooks()
    hooks.on("turn_start", lambda ev, ctx: TurnStartResult(system_prompt="OVERRIDDEN"))
    agent = Agent(client=client, model="gpt-4o", system_prompt="base", hooks=hooks)

    await agent.run("hi")

    system_msgs = [m for m in captured[0]["messages"] if m.get("role") == "system"]
    assert system_msgs and system_msgs[0]["content"] == "OVERRIDDEN"


# --- compaction -----------------------------------------------------------


async def test_before_compact_can_cancel() -> None:
    history = [
        UserMessage(content="a" * 500),
        AssistantMessage(content=[TextContent(text="b" * 500)], stop_reason="stop"),
        UserMessage(content="c" * 500),
    ]
    hooks = Hooks()
    hooks.on("before_compact", lambda ev, ctx: CompactResult(cancel=True))

    result = await compact(
        history,
        client=Client(),
        model="gpt-4o",
        keep_recent_tokens=10,
        hooks=hooks,
    )

    assert result is None


async def test_before_compact_receives_cut_index() -> None:
    history = [
        UserMessage(content="a" * 500),
        AssistantMessage(content=[TextContent(text="b" * 500)], stop_reason="stop"),
        UserMessage(content="c" * 500),
    ]
    seen: list[BeforeCompact] = []
    hooks = Hooks()
    hooks.on("before_compact", lambda ev, ctx: seen.append(ev) or CompactResult(cancel=True))

    await compact(
        history, client=Client(), model="gpt-4o", keep_recent_tokens=10, hooks=hooks
    )

    assert len(seen) == 1
    assert seen[0].cut_index > 0
