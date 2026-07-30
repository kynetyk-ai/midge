from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from midge.agent import Agent
from midge.client import Client
from midge.config import ProviderConfig
from midge.hooks import (
    Hooks,
    ProviderRequestResult,
    ToolCallEvent,
    ToolCallResult,
    TurnStartResult,
)
from midge.messages import AssistantMessage, TextContent, ToolCall, UserMessage
from midge.persistence import Session, read_transcript, session_continuations
from midge.providers import ModelRegistry
from midge.subagents import (
    SubagentTool,
    _ChildHooks,
    bind_subagents,
    subagent,
)
from midge.subagents import validate as subagent_validate
from midge.tools import Tool, ToolRegistry, tool
from tests.fakes import FakeProvider, ScriptedProvider, finish, install, say, tcall

PROMPT = "You are a test explorer."


def _says(text: str) -> list[Any]:
    return [say(text), finish()]


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
    session: Session | None = None,
    **bind_kw: Any,
) -> tuple[ToolRegistry, Client]:
    client = Client()
    install(client, turns)
    registry = ToolRegistry([read, write, *(extra or []), tool_obj])
    bind_subagents(
        registry,
        client=client,
        model="parent-model",
        hooks=hooks,
        session=session,
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
    install(client, [[say("done"), finish()]])
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
    captured = install(client, [_says("ok")])

    await registry.invoke("spawn_explore", {"question": "where?", "paths": ["a.py"]}, call_id="c1")

    sent = captured[0]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[0]["content"] == PROMPT
    assert "Question: where?" in sent[1]["content"]
    assert "a.py" in sent[1]["content"]


async def test_child_model_overrides_and_empty_inherits() -> None:
    registry, client = _bound(_explorer(model="child-model"), turns=[_says("ok")])
    captured = install(client, [_says("ok")])
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    assert captured[0]["model"] == "child-model"

    registry, client = _bound(_explorer(), turns=[_says("ok")])
    captured = install(client, [_says("ok")])
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    assert captured[0]["model"] == "parent-model"


async def test_child_registry_is_exactly_the_allowlist() -> None:
    registry, client = _bound(_explorer(), turns=[_says("ok")])
    captured = install(client, [_says("ok")])
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
        turns=[[say(""), finish("error")]],
    )
    out = await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    assert isinstance(out, str)
    assert "did not finish" in out


async def test_timeout_returns_a_tool_error_string() -> None:
    client = Client()

    async def on_open(body: Any) -> list[Any]:
        await asyncio.sleep(10)
        return []

    client.provider = ScriptedProvider(on_open)
    registry = ToolRegistry([read, _explorer(timeout=0.05)])
    bind_subagents(registry, client=client, model="m")

    out = await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    assert "timed out" in out


# ---- the timeout has three owners ----
#
# The author sets a budget for this agent; the caller may say this job is
# bigger; the operator caps the lot. The clamp is what makes the caller's say
# safe to offer, since "specify a timeout" would otherwise mean "specify none".


def _slow_agent(**kw: Any) -> SubagentTool:
    """An explorer that offers `timeout` in its signature — the opt-in."""
    opts: dict[str, Any] = {"description": "d", "prompt": PROMPT, "tools": ("read",)}
    opts.update(kw)

    @subagent(**opts)
    async def explore(question: str, timeout: float | None = None) -> str:
        return f"Question: {question}"

    return explore


def test_a_caller_timeout_appears_only_when_the_author_offers_it() -> None:
    # "The function signature is the tool schema" with no exception: an author
    # who wants a fixed budget simply does not declare the parameter.
    assert "timeout" in _slow_agent().schema()["parameters"]["properties"]
    assert "timeout" not in _explorer().schema()["parameters"]["properties"]


async def test_the_caller_can_ask_for_longer_than_the_author_set() -> None:
    client = Client()

    async def on_open(body: Any) -> list[Any]:
        await asyncio.sleep(0.2)
        return []

    client.provider = ScriptedProvider(on_open)
    registry = ToolRegistry([read, _slow_agent(timeout=0.05)])
    bind_subagents(registry, client=client, model="m")

    out = await registry.invoke(
        "spawn_explore", {"question": "q", "timeout": 5.0}, call_id="c1"
    )
    # Would have timed out at the author's 0.05s; the caller's 5s let it run.
    assert "timed out" not in out


async def test_the_operator_ceiling_clamps_the_caller() -> None:
    client = Client()

    async def on_open(body: Any) -> list[Any]:
        await asyncio.sleep(10)
        return []

    client.provider = ScriptedProvider(on_open)
    registry = ToolRegistry([read, _slow_agent(timeout=60)])
    bind_subagents(registry, client=client, model="m", max_timeout=0.05)

    out = await registry.invoke(
        "spawn_explore", {"question": "q", "timeout": 3600}, call_id="c1"
    )
    assert "timed out after 0s" in out


# ---- recursion is denied where it would close, and nowhere else ----
#
# No depth cap: an author who names a `spawn_*` tool has granted nesting on
# purpose, and a global number would only override a declaration sitting in
# their file. What must not happen is recursion with no end, so a child never
# receives a spawn tool for an agent already running above it. Termination
# follows anyway — the ancestor set grows one name per level from a finite
# declared set.


def _kids(t: SubagentTool) -> ToolRegistry:
    from midge.subagents import _child_registry

    assert t.runtime is not None
    return _child_registry(t.spec, t.runtime)


async def test_a_granted_spawn_tool_is_rebound_one_deeper() -> None:
    alpha = _explorer(name="alpha", tools=("read", "spawn_beta"))
    beta = _explorer(name="beta", tools=("read",))
    _bound(alpha, turns=[_says("ok")], extra=[beta])

    nested = _kids(alpha).get("spawn_beta")

    assert isinstance(nested, SubagentTool)
    assert nested is not beta, "the parent's tool must not be reused"
    assert nested.runtime is not None
    assert nested.runtime.depth == 1
    assert nested.runtime.ancestors == frozenset({"alpha"})
    assert alpha.runtime is not None and alpha.runtime.depth == 0


async def test_an_agent_cannot_spawn_itself() -> None:
    t = _explorer(tools=("read", "spawn_explore"))
    _bound(t, turns=[_says("ok")])

    child = _kids(t)
    assert "spawn_explore" not in child
    assert "read" in child, "the rest of the allowlist is untouched"


async def test_an_agent_running_above_is_excluded_but_survives_elsewhere() -> None:
    """The whole point of checking ancestry rather than editing the graph:
    `beta -> alpha` is denied only where alpha is on the stack."""
    alpha = _explorer(name="alpha", tools=("spawn_beta",))
    beta = _explorer(name="beta", tools=("spawn_alpha",))
    gamma = _explorer(name="gamma", tools=("spawn_alpha",))
    _bound(alpha, turns=[_says("ok")], extra=[beta, gamma])

    # alpha is running, so the beta beneath it cannot come back to alpha.
    under_alpha = _kids(alpha).get("spawn_beta")
    assert isinstance(under_alpha, SubagentTool)
    assert "spawn_alpha" not in _kids(under_alpha)

    # gamma is not alpha, so the same declaration works there.
    assert "spawn_alpha" in _kids(gamma)


async def test_an_allowlist_without_a_spawn_tool_cannot_recurse() -> None:
    t = _explorer(tools=("read", "bash"))
    _bound(t, turns=[_says("ok")])

    child = _kids(t)
    assert "spawn_explore" not in child
    assert "read" in child


def test_a_cyclic_allowlist_is_reported_but_costs_nothing() -> None:
    """A bug in the declaration worth telling the author about. Taking their
    agents away is not the same as telling them, and the recursion cannot
    happen regardless."""
    a = _explorer(name="alpha", tools=("spawn_beta",))
    b = _explorer(name="beta", tools=("spawn_alpha",))
    registry = ToolRegistry([a, b])

    diagnostics = subagent_validate(registry)

    assert [d.event for d in diagnostics] == ["subagent_cycle"]
    assert "alpha -> beta -> alpha" in diagnostics[0].fields["agents"]
    assert len(registry) == 2, "both agents still load"


def test_a_self_reference_is_reported_too() -> None:
    t = _explorer(tools=("read", "spawn_explore"))
    registry = ToolRegistry([read, t])

    diagnostics = subagent_validate(registry)

    assert [d.event for d in diagnostics] == ["subagent_cycle"]
    assert "spawn_explore" in registry


def test_an_acyclic_chain_says_nothing() -> None:
    # A -> B -> C is a design, not a mistake.
    a = _explorer(name="alpha", tools=("spawn_beta",))
    b = _explorer(name="beta", tools=("spawn_gamma",))
    c = _explorer(name="gamma", tools=("read",))
    registry = ToolRegistry([read, a, b, c])

    assert subagent_validate(registry) == []


def test_an_unknown_tool_name_is_a_warning_not_a_death() -> None:
    """Nothing checked this before, so `tools=("raed",)` silently yielded a
    smaller child registry. A typo in one name should not cost the agent."""
    t = _explorer(tools=("read", "raed"))
    registry = ToolRegistry([read, t])

    diagnostics = subagent_validate(registry)

    assert [d.event for d in diagnostics] == ["subagent_tool_unknown"]
    assert diagnostics[0].fields["tool"] == "raed"
    assert "spawn_explore" in registry


def test_a_declared_model_is_checked_against_the_registry() -> None:
    """It used to fail at the first delegation, inside a turn, as the vendor's
    404 dressed up as a tool result. Dropped rather than warned: unlike a cycle
    or a typo, this agent cannot run at all."""
    t = _explorer(model="gtp-4o")
    registry = ToolRegistry([read, t])
    models = ModelRegistry(
        models={"gpt-4o": "p"}, providers={"p": ProviderConfig(kind="openai")}
    )

    diagnostics = subagent_validate(registry, models=models)

    assert [d.event for d in diagnostics] == ["subagent_model_unregistered"]
    assert "spawn_explore" not in registry
    assert "read" in registry, "an unrelated tool is untouched"


def test_an_empty_model_registry_is_permissive() -> None:
    t = _explorer(model="anything-at-all")
    registry = ToolRegistry([read, t])

    assert subagent_validate(registry) == []
    assert "spawn_explore" in registry


# ---- concurrency ----


async def test_semaphore_caps_concurrent_children() -> None:
    live = 0
    peak = 0

    client = Client()

    async def on_open(body: Any) -> list[Any]:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return _says("ok")

    client.provider = ScriptedProvider(on_open)
    registry = ToolRegistry([read, _explorer()])
    bind_subagents(registry, client=client, model="m", max_concurrent=2)

    await asyncio.gather(
        *(registry.invoke("spawn_explore", {"question": str(i)}, call_id=f"c{i}") for i in range(6))
    )
    assert peak <= 2


async def test_parent_cancellation_reaches_the_child() -> None:
    cancelled = asyncio.Event()

    client = Client()

    async def on_open(body: Any) -> list[Any]:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return []

    client.provider = ScriptedProvider(on_open)
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
                tcall(index=0, id="t1", name="read", args='{"path":"x"}'),
                finish("tool_use"),
            ],
            _says("could not read"),
        ],
        hooks=hooks,
    )
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")

    assert blocked == ["read"]


async def test_child_tool_results_reach_the_parent_policy() -> None:
    seen: list[str] = []
    hooks = Hooks()
    hooks.on("tool_result", lambda e, c: seen.append(e.tool_call.name))

    registry, _ = _bound(
        _explorer(),
        turns=[
            [
                tcall(index=0, id="t1", name="read", args='{"path":"x"}'),
                finish("tool_use"),
            ],
            _says("done"),
        ],
        hooks=hooks,
    )
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    assert seen == ["read"]


async def test_observers_still_see_the_child_tool_calls() -> None:
    """`approve.py` audits via observe(); narrowing must not blind it."""
    seen: list[str] = []

    def audit(event: Any, ctx: Any) -> None:
        if event.type == "tool_call":
            seen.append(event.tool_call.name)

    hooks = Hooks()
    hooks.observe(audit)

    registry, _ = _bound(
        _explorer(),
        turns=[
            [
                tcall(index=0, id="t1", name="read", args='{"path":"x"}'),
                finish("tool_use"),
            ],
            _says("done"),
        ],
        hooks=hooks,
    )
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    assert seen == ["read"]


async def test_parent_turn_start_cannot_replace_the_child_prompt() -> None:
    """The child's prompt is the whole point of specialising it — a handler
    written for the parent conversation must not silently override it."""
    hooks = Hooks()
    hooks.on("turn_start", lambda e, c: TurnStartResult(system_prompt="PARENT-OVERRIDE"))

    registry, client = _bound(_explorer(), turns=[_says("ok")], hooks=hooks)
    captured = install(client, [_says("ok")])
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")

    assert captured[0]["messages"][0]["content"] == PROMPT


async def test_parent_request_hook_cannot_relabel_the_child_tools() -> None:
    hooks = Hooks()
    hooks.on(
        "before_provider_request",
        lambda e, c: ProviderRequestResult(
            tools=[{"name": "write", "description": "w", "parameters": {}}]
        ),
    )

    registry, client = _bound(_explorer(), turns=[_says("ok")], hooks=hooks)
    captured = install(client, [_says("ok")])
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")

    names = {t["function"]["name"] for t in captured[0]["tools"]}
    assert names == {"read"}


async def test_parent_turn_hooks_still_fire_for_the_parent() -> None:
    """Narrowing applies to the child only; the parent keeps every event."""
    fired: list[str] = []
    hooks = Hooks()
    hooks.on("turn_start", lambda e, c: fired.append("turn_start"))
    hooks.on("turn_end", lambda e, c: fired.append("turn_end"))

    client = Client()
    install(client, [_says("hi")])
    agent = Agent(client=client, model="m", tools=ToolRegistry([read]), hooks=hooks)
    await agent.run("hello")

    assert fired == ["turn_start", "turn_end"]


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

    registry, _ = _bound(_explorer(), turns=[_says("found it")], session=parent)
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
    parent = Session.new(parent_path, model="m")

    registry, _ = _bound(_explorer(), turns=[_says("found it")], session=parent)
    await registry.invoke("spawn_explore", {"question": "where?"}, call_id="c1")
    parent.close()

    child_path = next(p for p in tmp_path.iterdir() if p != parent_path)
    messages = Session.load(child_path).messages
    assert isinstance(messages[0], UserMessage)
    assert "Question: where?" in str(messages[0].content)
    assert any(isinstance(m, AssistantMessage) for m in messages)


async def test_no_transcript_without_a_parent_session(tmp_path: Path) -> None:
    registry, _ = _bound(_explorer(), turns=[_says("ok")], session=None)
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    assert list(tmp_path.iterdir()) == []


async def test_hostile_call_id_is_sanitised(tmp_path: Path) -> None:
    parent_path = tmp_path / "run.jsonl"
    parent = Session.new(parent_path, model="m")

    registry, _ = _bound(_explorer(), turns=[_says("ok")], session=parent)
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="../../etc/passwd")
    parent.close()

    children = [p for p in tmp_path.iterdir() if p != parent_path]
    assert len(children) == 1
    assert children[0].parent == tmp_path
    assert "/" not in children[0].name


async def test_empty_call_id_falls_back_to_an_index(tmp_path: Path) -> None:
    parent_path = tmp_path / "run.jsonl"
    parent = Session.new(parent_path, model="m")

    registry, _ = _bound(_explorer(), turns=[_says("a"), _says("b")], session=parent)
    await registry.invoke("spawn_explore", {"question": "q"}, call_id=None)
    await registry.invoke("spawn_explore", {"question": "q"}, call_id=None)
    parent.close()

    children = sorted(p.name for p in tmp_path.iterdir() if p != parent_path)
    assert children == ["run.explore-0.jsonl", "run.explore-1.jsonl"]


async def test_a_child_transcript_says_it_is_a_subagent_run(tmp_path: Path) -> None:
    """Before `origin`, this was identifiable only by accident — the file
    happened to have `parent_tool_call_id` set."""
    parent_path = tmp_path / "run.jsonl"
    parent = Session.new(parent_path, model="m")

    registry, _ = _bound(_explorer(), turns=[_says("ok")], session=parent)
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    parent.close()

    child_path = next(p for p in tmp_path.iterdir() if p != parent_path)
    assert Session.load(child_path).header.origin == "subagent"


async def test_the_parent_records_a_forward_link_to_the_child(tmp_path: Path) -> None:
    """`parent_session` is a back-pointer, so without this "which transcripts
    belong to this run" is a directory scan."""
    parent_path = tmp_path / "run.jsonl"
    parent = Session.new(parent_path, model="m")

    registry, _ = _bound(_explorer(), turns=[_says("ok")], session=parent)
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    parent.close()

    child_path = next(p for p in tmp_path.iterdir() if p != parent_path)
    _header, entries = read_transcript(parent_path)
    (link,) = session_continuations(entries)
    assert link.reason == "subagent"
    # Relative to the parent's directory, not absolute: an audit trail should
    # not bake one machine's layout in, and children are always siblings.
    assert link.path == child_path.name
    assert (parent_path.parent / link.path) == child_path


async def test_the_chain_walks_both_ways(tmp_path: Path) -> None:
    parent_path = tmp_path / "run.jsonl"
    parent = Session.new(parent_path, model="m")

    registry, _ = _bound(_explorer(), turns=[_says("ok")], session=parent)
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    parent.close()

    _header, entries = read_transcript(parent_path)
    (link,) = session_continuations(entries)
    child = Session.load(parent_path.parent / link.path)
    assert child.header.parent_session == str(parent_path)


async def test_every_child_gets_its_own_forward_link(tmp_path: Path) -> None:
    parent_path = tmp_path / "run.jsonl"
    parent = Session.new(parent_path, model="m")

    registry, _ = _bound(_explorer(), turns=[_says("a"), _says("b")], session=parent)
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")
    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c2")
    parent.close()

    _header, entries = read_transcript(parent_path)
    links = session_continuations(entries)
    assert sorted(r.path for r in links) == sorted(
        p.name for p in tmp_path.iterdir() if p != parent_path
    )


async def test_no_forward_link_without_a_parent_session(tmp_path: Path) -> None:
    registry, _ = _bound(_explorer(), turns=[_says("ok")], session=None)
    result = await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")

    # The delegation still runs and still answers; only the transcript is absent.
    assert "ok" in str(result)
    assert list(tmp_path.iterdir()) == []


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
    install(client, [_says("done")])
    agent = Agent(client=client, model="m", tools=ToolRegistry([probe]))
    await agent._run_tool(ToolCall(id="call_xyz", name="probe", arguments={"path": "a"}))

    assert seen == ["call_xyz"]


async def test_plain_tools_are_unaffected_by_call_id() -> None:
    registry = ToolRegistry([read])
    assert await registry.invoke("read", {"path": "a"}, call_id="c1") == "contents of a"
    assert await registry.invoke("read", {"path": "a"}) == "contents of a"


async def test_a_child_on_another_provider_routes_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sub-agent half of #71.

    A child shares the parent's `Client` — `subagents.py` passes
    `client=runtime.client, model=spec.model or runtime.model` — so before the
    registry, `@subagent(model=...)` only worked when the child's model happened
    to live on the parent's provider. Now the model chooses, per call.
    """
    from midge.config import ProviderConfig
    from midge.providers import ModelRegistry

    # Keyed by adapter kind, which is what `get` resolves.
    built: dict[str, FakeProvider] = {}

    def fake_get(kind: str) -> Any:
        def factory(**_: Any) -> FakeProvider:
            provider = FakeProvider([_says("ok")])
            built[kind] = provider
            return provider

        return factory

    monkeypatch.setattr("midge.providers.registry.get", fake_get)
    client = Client(
        registry=ModelRegistry(
            models={"parent-model": "up", "child-model": "down"},
            providers={
                "up": ProviderConfig(kind="upstream"),
                "down": ProviderConfig(kind="downstream"),
            },
        )
    )
    registry = ToolRegistry([_explorer(model="child-model")])
    bind_subagents(registry, client=client, model="parent-model")

    await registry.invoke("spawn_explore", {"question": "q"}, call_id="c1")

    # Only the child ran here, and it went to its own provider — the parent's
    # was never touched.
    assert set(built) == {"downstream"}
    assert built["downstream"].bodies[0]["model"] == "child-model"


async def test_a_child_inherits_the_parent_active_source_set() -> None:
    """Deactivating a source in the parent has to reach delegated work too.

    `_ChildHooks` delegates to the parent's `emit`, which is where activation
    filters — so this falls out of the design rather than needing its own
    mechanism. Pinned because the opposite would be a hole: an approval hook
    switched off for the parent but still firing for its sub-agents would make
    a profile's hook list mean two different things at two depths.
    """
    parent = Hooks()
    parent.on(
        "tool_call",
        lambda event, ctx: ToolCallResult(block=True, reason="denied"),
        name="approve",
    )
    child = _ChildHooks(parent)
    call = ToolCall(id="c1", name="read", arguments={})

    assert await child.emit(ToolCallEvent(tool_call=call)) is not None

    parent.set_active_sources(set())

    assert await child.emit(ToolCallEvent(tool_call=call)) is None
