from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from midge.agent import Agent
from midge.client import Client
from midge.hooks import Hooks, ToolCallResult
from midge.messages import AssistantMessage, TextContent, ToolCall, UserMessage
from midge.persistence import Session
from midge.subagents import (
    SubagentTool,
    bind_subagents,
    subagent,
)
from midge.tools import Tool, ToolRegistry, tool

PROMPT = "You are a test explorer."


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


def _tcd(*, index: int, id: str, name: str, arguments: str) -> Any:
    return SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )


class _FakeStream:
    def __init__(self, chunks: Iterable[Any]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> Any:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _install(client: Client, turns: list[list[Any]]) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    iterator = iter(turns)

    async def create(**kwargs: Any) -> _FakeStream:
        captured.append(kwargs)
        return _FakeStream(next(iterator))

    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return captured


def _says(text: str) -> list[Any]:
    return [_chunk(content=text), _chunk(finish_reason="stop")]


@tool
async def read(path: str) -> str:
    """Read a file."""
    return f"contents of {path}"


@tool
async def write(path: str, content: str) -> str:
    """Write a file."""
    return "ok"


def _explorer(**kw: Any) -> SubagentTool:
    opts: dict[str, Any] = {
        "description": "Explore the codebase.",
        "prompt": PROMPT,
        "tools": ("read",),
    }
    opts.update(kw)

    @subagent(**opts)
    async def explore(question: str, paths: list[str] | None = None) -> str:
        """Compose the opening message."""
        scope = "\n".join(paths or ["(all)"])
        return f"Question: {question}\n\nScope:\n{scope}"

    return explore


def _bound(
    tool_obj: SubagentTool,
    *,
    turns: list[list[Any]],
    extra: list[Tool] | None = None,
    hooks: Hooks | None = None,
    session_path: Path | None = None,
    **bind_kw: Any,
) -> tuple[ToolRegistry, Client]:
    client = Client()
    _install(client, turns)
    registry = ToolRegistry([read, write, *(extra or []), tool_obj])
    bind_subagents(
        registry,
        client=client,
        model="parent-model",
        hooks=hooks,
        session_path=session_path,
        **bind_kw,
    )
    return registry, client


# ---- declaration ----


def test_subagent_produces_a_spawn_prefixed_tool() -> None:
    t = _explorer()
    assert isinstance(t, SubagentTool)
    assert t.name == "spawn_explore"
    assert t.description == "Explore the codebase."


def test_signature_becomes_the_tool_schema() -> None:
    params = _explorer().schema()["parameters"]
    assert set(params["properties"]) == {"question", "paths"}
    assert params["required"] == ["question"]
    assert params["additionalProperties"] is False


def test_unknown_argument_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _explorer().params_model.model_validate({"question": "q", "nope": 1})


def test_explicit_name_overrides() -> None:
    assert _explorer(name="scout").name == "spawn_scout"


def test_sync_function_is_rejected() -> None:
    with pytest.raises(TypeError, match="async"):

        @subagent(description="d", prompt="p")  # type: ignore[arg-type]
        def broken(question: str) -> str:
            return question


# ---- binding ----


async def test_unbound_tool_reports_the_fix() -> None:
    with pytest.raises(RuntimeError, match="bind_subagents"):
        await _explorer().invoke({"question": "q"})


async def test_unbound_tool_becomes_a_tool_error_not_a_crash() -> None:
    client = Client()
    _install(client, [[_chunk(content="done"), _chunk(finish_reason="stop")]])
    agent = Agent(client=client, model="m", tools=ToolRegistry([_explorer()]))
    result = await agent._run_tool(ToolCall(id="c1", name="spawn_explore", arguments={"question": "q"}))

    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "bind_subagents" in result.content[0].text


def test_bind_is_a_no_op_without_subagents(caplog: pytest.LogCaptureFixture) -> None:
    registry = ToolRegistry([read])
    with caplog.at_level(logging.INFO, logger="midge.subagents"):
        bind_subagents(registry, client=Client(), model="m")
    assert caplog.records == []


# ---- running the child ----


async def test_parent_receives_only_the_final_text() -> None:
    registry, _ = _bound(_explorer(), turns=[_says("found it at foo.py:12")])
    out = await registry.invoke("spawn_explore", {"question": "where?"}, call_id="c1")
    assert out == "found it at foo.py:12"


async def test_opening_message_comes_from_the_function() -> None:
    t = _explorer()
    registry, client = _bound(t, turns=[_says("ok")])
    captured = _install(client, [_says("ok")])

    await registry.invoke("spawn_explore", {"question": "where?", "paths": ["a.py"]}, call_id="c1")

    sent = captured[0]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[0]["content"] == PROMPT
    assert "Question: where?" in sent[1]["content"]
    assert "a.py" in sent[1]["content"]


async def test_child_model_overrides_and_empty_inherits() -> None:
    registry, client = _bound(_explorer(model="child-model"), turns=[_says("ok")])
    captured = _install(client, [_says("ok")])
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    assert captured[0]["model"] == "child-model"

    registry, client = _bound(_explorer(), turns=[_says("ok")])
    captured = _install(client, [_says("ok")])
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    assert captured[0]["model"] == "parent-model"


async def test_child_registry_is_exactly_the_allowlist() -> None:
    registry, client = _bound(_explorer(), turns=[_says("ok")])
    captured = _install(client, [_says("ok")])
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")

    names = {t["function"]["name"] for t in captured[0]["tools"]}
    assert names == {"read"}  # `write` and `spawn_explore` are not allowlisted


def test_allowlisted_tool_is_the_same_object() -> None:
    from midge.subagents import _child_registry

    t = _explorer()
    _bound(t, turns=[])
    assert t.runtime is not None

    child = _child_registry(t.spec, t.runtime)
    assert child.get("read") is read
    assert "write" not in child


async def test_child_error_becomes_a_readable_string() -> None:
    registry, _ = _bound(
        _explorer(),
        turns=[[_chunk(content=""), _chunk(finish_reason="content_filter")]],
    )
    out = await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    assert isinstance(out, str)
    assert "did not finish" in out


async def test_timeout_returns_a_tool_error_string() -> None:
    client = Client()

    async def create(**kwargs: Any) -> _FakeStream:
        await asyncio.sleep(10)
        return _FakeStream([])

    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    registry = ToolRegistry([read, _explorer(timeout=0.05)])
    bind_subagents(registry, client=client, model="m")

    out = await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    assert "timed out" in out


# ---- depth ----


async def test_child_gets_spawn_tools_rebound_one_deeper() -> None:
    from midge.subagents import _child_registry

    t = _explorer(tools=("read", "spawn_explore"))
    _bound(t, turns=[_says("ok")], max_depth=2)
    assert t.runtime is not None

    child = _child_registry(t.spec, t.runtime)
    nested = child.get("spawn_explore")

    assert isinstance(nested, SubagentTool)
    assert nested is not t, "the parent's tool must not be reused"
    assert nested.runtime is not None
    assert nested.runtime.depth == 1
    assert t.runtime.depth == 0, "the parent's runtime must be untouched"


async def test_spawn_tools_vanish_at_max_depth() -> None:
    from midge.subagents import _child_registry

    t = _explorer(tools=("read", "spawn_explore"))
    _bound(t, turns=[_says("ok")], max_depth=1)
    assert t.runtime is not None

    child = _child_registry(t.spec, t.runtime)
    assert "spawn_explore" not in child
    assert "read" in child


# ---- concurrency ----


async def test_semaphore_caps_concurrent_children() -> None:
    live = 0
    peak = 0

    client = Client()

    async def create(**kwargs: Any) -> _FakeStream:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return _FakeStream(_says("ok"))

    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    registry = ToolRegistry([read, _explorer()])
    bind_subagents(registry, client=client, model="m", max_concurrent=2)

    await asyncio.gather(
        *(registry.invoke("spawn_explore", {"question": str(i)}, call_id=f"c{i}") for i in range(6))
    )
    assert peak <= 2


async def test_parent_cancellation_reaches_the_child() -> None:
    cancelled = asyncio.Event()

    client = Client()

    async def create(**kwargs: Any) -> _FakeStream:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return _FakeStream([])

    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    registry = ToolRegistry([read, _explorer()])
    bind_subagents(registry, client=client, model="m")

    task = asyncio.ensure_future(registry.invoke("spawn_explore", {"question": "q"}, call_id="c1"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set(), "the child stream was abandoned, not cancelled"


# ---- hooks ----


async def test_parent_policy_applies_to_the_child() -> None:
    """Without shared hooks, delegation would be a way around an approval policy."""
    blocked: list[str] = []

    def deny(event: Any, ctx: Any) -> Any:
        blocked.append(event.tool_call.name)
        return ToolCallResult(block=True, reason="denied by policy")

    hooks = Hooks()
    hooks.on("tool_call", deny)

    registry, _ = _bound(
        _explorer(),
        turns=[
            [
                _chunk(
                    tool_calls=[_tcd(index=0, id="t1", name="read", arguments='{"path":"x"}')]
                ),
                _chunk(finish_reason="tool_calls"),
            ],
            _says("could not read"),
        ],
        hooks=hooks,
    )
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")

    assert blocked == ["read"]


# ---- transcripts ----


async def test_transcript_links_to_the_parent_tool_call(tmp_path: Path) -> None:
    parent_path = tmp_path / "run.jsonl"
    parent = Session.new(parent_path, model="parent-model")
    parent.append(UserMessage(content="where?"))
    parent.append(
        AssistantMessage(
            content=[ToolCall(id="call_abc123", name="spawn_explore", arguments={"question": "q"})],
            stop_reason="tool_use",
        )
    )

    registry, _ = _bound(_explorer(), turns=[_says("found it")], session_path=parent_path)
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="call_abc123")
    parent.close()

    children = [p for p in tmp_path.iterdir() if p != parent_path]
    assert len(children) == 1
    child_path = children[0]

    # The link is by id, not by filename convention: load both and match.
    child = Session.load(child_path)
    assert child.header.parent_tool_call_id == "call_abc123"
    assert child.header.parent_session == str(parent_path)

    parent_ids = {
        c.id
        for m in Session.load(parent_path).messages
        if isinstance(m, AssistantMessage)
        for c in m.content
        if isinstance(c, ToolCall)
    }
    assert child.header.parent_tool_call_id in parent_ids
    assert "call_abc123" in child_path.name


async def test_transcript_holds_the_child_history(tmp_path: Path) -> None:
    parent_path = tmp_path / "run.jsonl"
    Session.new(parent_path, model="m").close()

    registry, _ = _bound(_explorer(), turns=[_says("found it")], session_path=parent_path)
    await registry.invoke("spawn_explore", {"question": "where?"}, call_id="c1")

    child_path = next(p for p in tmp_path.iterdir() if p != parent_path)
    messages = Session.load(child_path).messages
    assert isinstance(messages[0], UserMessage)
    assert "Question: where?" in str(messages[0].content)
    assert any(isinstance(m, AssistantMessage) for m in messages)


async def test_no_transcript_without_a_parent_session(tmp_path: Path) -> None:
    registry, _ = _bound(_explorer(), turns=[_says("ok")], session_path=None)
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    assert list(tmp_path.iterdir()) == []


async def test_hostile_call_id_is_sanitised(tmp_path: Path) -> None:
    parent_path = tmp_path / "run.jsonl"
    Session.new(parent_path, model="m").close()

    registry, _ = _bound(_explorer(), turns=[_says("ok")], session_path=parent_path)
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="../../etc/passwd")

    children = [p for p in tmp_path.iterdir() if p != parent_path]
    assert len(children) == 1
    assert children[0].parent == tmp_path
    assert "/" not in children[0].name


async def test_empty_call_id_falls_back_to_an_index(tmp_path: Path) -> None:
    parent_path = tmp_path / "run.jsonl"
    Session.new(parent_path, model="m").close()

    registry, _ = _bound(_explorer(), turns=[_says("a"), _says("b")], session_path=parent_path)
    await registry.invoke("spawn_explore", {"question": "q"}, call_id=None)
    await registry.invoke("spawn_explore", {"question": "q"}, call_id=None)

    children = sorted(p.name for p in tmp_path.iterdir() if p != parent_path)
    assert children == ["run.explore-0.jsonl", "run.explore-1.jsonl"]


def test_header_fields_are_additive(tmp_path: Path) -> None:
    """A session written before these fields existed must still load."""
    path = tmp_path / "old.jsonl"
    path.write_text(
        '{"type":"header","version":1,"created_at":"2026-01-01","model":"m"}\n',
        encoding="utf-8",
    )
    header = Session.load(path).header
    assert header.parent_session is None
    assert header.parent_tool_call_id is None


def test_header_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "child.jsonl"
    Session.new(
        path, model="m", parent_session="/tmp/parent.jsonl", parent_tool_call_id="call_1"
    ).close()

    raw = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert raw["parent_tool_call_id"] == "call_1"


# ---- call_id plumbing ----


async def test_call_id_reaches_tool_invoke() -> None:
    seen: list[str | None] = []

    class _Probe(Tool):
        async def invoke(self, arguments: dict[str, Any], *, call_id: str | None = None) -> Any:
            seen.append(call_id)
            return "ok"

    probe = _Probe(
        name="probe", description="d", fn=read.fn, params_model=read.params_model
    )
    client = Client()
    _install(client, [_says("done")])
    agent = Agent(client=client, model="m", tools=ToolRegistry([probe]))
    await agent._run_tool(ToolCall(id="call_xyz", name="probe", arguments={"path": "a"}))

    assert seen == ["call_xyz"]


async def test_plain_tools_are_unaffected_by_call_id() -> None:
    registry = ToolRegistry([read])
    assert await registry.invoke("read", {"path": "a"}, call_id="c1") == "contents of a"
    assert await registry.invoke("read", {"path": "a"}) == "contents of a"
