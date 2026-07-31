"""Sub-agents: predefined nested agents, each surfaced as its own tool.

A sub-agent is a `spawn_<name>` tool. That is the whole design, and several
things fall out of it: the agent list needs no system-prompt catalogue because
tools are already self-describing; an invalid agent name is unrepresentable
because the registry enforces it; the `tool_call` hook can block one agent by
name; and recursion control is the same mechanism as tool subsetting, since a
child only has `spawn_*` tools if its own allowlist names them.

There is no depth cap: the allowlist is the control, and a global number would
override a declaration in the author's file. Recursion is instead denied where
it would close — a child gets no `spawn_*` tool for an agent already running
above it, so `alpha -> beta` always works and `beta -> alpha` is refused only
when alpha is on the stack. That also guarantees termination, since the ancestor
set grows one name per level from a finite declared set. `validate` warns about
a cyclic allowlist rather than refusing it.

The `.py` file declares what the agent is for *and* the inputs midge must
supply. The decorated function's signature becomes the tool schema and its
return value becomes the child's opening message, so the model fills in
declared, validated fields and cannot hand a sub-agent an arbitrary system
prompt, tool list, or model:

    @subagent(
        description="Locate where something lives. Read-only.",
        prompt=EXPLORE_PROMPT,
        tools=("read", "bash"),
    )
    async def explore(question: str, paths: list[str] | None = None) -> str:
        scope = "\\n".join(paths or ["(whole repository)"])
        return f"Question: {question}\\n\\nSearch scope:\\n{scope}"

Because `@subagent` returns a `Tool`, `load_extensions` already discovers it —
`--extension-dir` works unchanged and there is no separate loader.

Tools cannot see the agent that called them, so the entrypoint calls
`bind_subagents(registry, client=..., model=...)` once at startup to supply the
runtime. Only the parent's history sees the child's final text; the child's own
turns go to a sibling transcript keyed on the tool call id.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from midge.agent import Agent, AgentEnd, ToolExecutionEnd, ToolExecutionStart
from midge.client import Client, Done, Error
from midge.config import Diagnostic, SubagentConfig
from midge.hooks import Hooks
from midge.messages import AssistantMessage, TextContent, ToolCall
from midge.persistence import Session
from midge.providers import ModelRegistry
from midge.tools import Tool, ToolFn, ToolRegistry, _build_params_model

SubagentEvent = Callable[[Any, dict[str, Any]], Awaitable[None]]

# Enough to see a delegation happening. Deltas are excluded because a child
# emits hundreds per turn and its full stream is in its own transcript.
FORWARDED_EVENTS = (
    ToolExecutionStart,
    ToolExecutionEnd,
    Error,
    AgentEnd,
)

_SPAWN_PREFIX = "spawn_"
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SubagentSpec:
    name: str
    prompt: str
    tools: tuple[str, ...] = ()
    model: str = ""
    # None means "whatever `[subagents] timeout` says". A concrete default here
    # would make the config key unreachable, the same trap an argparse
    # `default=` sets.
    timeout: float | None = None


@dataclass(frozen=True, slots=True)
class SubagentRuntime:
    client: Client
    model: str
    registry: ToolRegistry
    limit: asyncio.Semaphore
    hooks: Hooks | None = None
    # The parent's open `Session`, not its path: only that handle may write to
    # the file, and a second one would append past its position so the parent's
    # next write overwrites the record.
    session: Session | None = None
    # For the logs. It bounds nothing — `ancestors` does.
    depth: int = 0
    # The agents running above this one. A child gets no `spawn_*` tool for any
    # of them, so recursion is denied where it would close and nowhere else.
    # Also what guarantees termination: the set grows one name per level from a
    # finite declared set.
    ancestors: frozenset[str] = frozenset()
    timeout: float = 300.0
    max_timeout: float = 900.0
    # The tool call that spawned this runtime's agent, None at the top level.
    # The same id the transcript records as `parent_tool_call_id`, so the wire
    # and the session name a run identically.
    agent_id: str | None = None
    # Where a nested agent's events go: raw events plus an envelope, never
    # wire-shaped. `event_to_wire` stays the one mapping layer, in `rpc.py`.
    on_event: SubagentEvent | None = None


# A child inherits tool policy and nothing else. `tool_call` and `tool_result`
# must apply recursively or delegation is a way around an approval policy. The
# rest are dropped because they were written with the parent's conversation in
# mind: a `turn_start` handler that sets `system_prompt` would silently replace
# the child's specialised prompt, and a `before_provider_request` handler that
# sets `tools` would advertise the child tools its registry will then refuse.
INHERITED_EVENTS = frozenset({"tool_call", "tool_result"})


class _ChildHooks(Hooks):
    """The parent's hooks, narrowed to what a nested agent should inherit.

    Filtering in `emit` rather than at registration covers `observe()` handlers
    too — they see every event that gets emitted, so an audit observer still
    sees the child's tool calls.
    """

    def __init__(self, parent: Hooks) -> None:
        super().__init__(parent.context)
        self._parent = parent

    async def emit(self, event: Any) -> Any | None:
        if event.type not in INHERITED_EVENTS:
            return None
        return await self._parent.emit(event)


class SubagentTool(Tool):
    """A tool whose function composes the child's opening message rather than
    doing the work. `invoke` runs the nested agent and returns its final text.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        fn: ToolFn,
        params_model: Any,
        spec: SubagentSpec,
    ) -> None:
        super().__init__(name=name, description=description, fn=fn, params_model=params_model)
        self.spec = spec
        self.runtime: SubagentRuntime | None = None

    def rebound(self, runtime: SubagentRuntime) -> SubagentTool:
        """A copy bound to `runtime`.

        Copy rather than mutate: the parent's tools are shared objects, and a
        child bound one level deeper must not change the parent's depth.
        """
        clone = SubagentTool(
            name=self.name,
            description=self.description,
            fn=self.fn,
            params_model=self.params_model,
            spec=self.spec,
        )
        clone.runtime = runtime
        return clone

    async def invoke(self, arguments: dict[str, Any], *, call_id: str | None = None) -> Any:
        if self.runtime is None:
            raise RuntimeError(
                f"{self.name} has no runtime; the entrypoint must call "
                "midge.subagents.bind_subagents(registry, client=..., model=...)"
            )
        validated = self.params_model.model_validate(arguments)
        kwargs = {f: getattr(validated, f) for f in self.params_model.model_fields}
        opening = await self.fn(**kwargs)
        # Opt-in: `timeout` reaches the schema only because the author put it
        # in the signature, so "the signature is the tool schema" holds.
        asked = kwargs.get("timeout")
        return await _run(
            self.spec,
            str(opening),
            self.runtime,
            call_id,
            asked_timeout=asked if isinstance(asked, int | float) else None,
        )


def subagent(
    *,
    description: str,
    prompt: str,
    tools: tuple[str, ...] = (),
    model: str = "",
    timeout: float | None = None,
    name: str | None = None,
) -> Callable[[ToolFn], SubagentTool]:
    """Declare a sub-agent. The decorated function's signature is the tool's
    schema; what it returns is the child's opening message.
    """

    def wrap(fn: ToolFn) -> SubagentTool:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"@subagent requires an async function; {fn.__name__} is not `async def`"
            )
        agent_name = name or fn.__name__
        tool_name = f"{_SPAWN_PREFIX}{agent_name}"
        return SubagentTool(
            name=tool_name,
            description=description,
            fn=fn,
            params_model=_build_params_model(fn, tool_name),
            spec=SubagentSpec(
                name=agent_name,
                prompt=prompt,
                tools=tuple(tools),
                model=model,
                timeout=timeout,
            ),
        )

    return wrap


def bind_subagents(
    registry: ToolRegistry,
    *,
    client: Client,
    model: str,
    hooks: Hooks | None = None,
    session: Session | None = None,
    subagents: SubagentConfig | None = None,
    on_event: SubagentEvent | None = None,
) -> None:
    """Give every `SubagentTool` in `registry` what it needs to run a child.

    A no-op on a registry with no sub-agents, so entrypoints can call it
    unconditionally. Takes the config section rather than loose limits, the way
    `logs.configure` takes a `LogConfig` — three parameters that must be kept in
    step is three chances to forget one.
    """
    limits = subagents if subagents is not None else SubagentConfig()
    runtime = SubagentRuntime(
        client=client,
        model=model,
        registry=registry,
        limit=asyncio.Semaphore(max(1, limits.max_concurrent)),
        hooks=hooks,
        session=session,
        depth=0,
        timeout=limits.timeout,
        max_timeout=limits.max_timeout,
        on_event=on_event,
    )
    bound = 0
    for t in registry:
        if isinstance(t, SubagentTool):
            t.runtime = runtime
            bound += 1
    if bound:
        _logger.info(
            "subagents_bound count=%d max_concurrent=%d timeout=%.0f max_timeout=%.0f",
            bound,
            limits.max_concurrent,
            limits.timeout,
            limits.max_timeout,
        )


def validate(
    registry: ToolRegistry, *, models: ModelRegistry | None = None
) -> list[Diagnostic]:
    """Report what is wrong with the declared sub-agents; drop only what cannot run.

    Run once after every source is loaded, for the reason profile validation is:
    one agent's allowlist may name another declared in a different file, and no
    ordering makes per-file checking correct.

    An **unregistered model** drops the agent, matching profiles: it would
    otherwise fail mid-turn as the vendor's 404 in a tool result. Empty registry
    stays permissive.

    An **unknown tool name** and a **cyclic allowlist** are warnings only. A
    typo in one name should not cost the whole agent, and a cycle cannot recurse
    anyway — `_child_registry` denies it where it would close.
    """
    diagnostics: list[Diagnostic] = []
    agents = {t.name: t for t in registry if isinstance(t, SubagentTool)}

    for name, tool_obj in sorted(agents.items()):
        spec = tool_obj.spec
        if models and spec.model and spec.model not in models:
            registry.remove(name)
            diagnostics.append(
                Diagnostic(
                    "subagent_model_unregistered",
                    {
                        "agent": spec.name,
                        "model": spec.model,
                        "registered": ",".join(models.names()),
                    },
                )
            )
            continue
        diagnostics.extend(
            Diagnostic("subagent_tool_unknown", {"agent": spec.name, "tool": wanted})
            for wanted in spec.tools
            if wanted not in registry
        )

    diagnostics.extend(
        Diagnostic(
            "subagent_cycle",
            {"agents": " -> ".join(agents[n].spec.name for n in [*cycle, cycle[0]])},
        )
        for cycle in _cycles(agents)
    )
    return diagnostics


def _cycles(agents: dict[str, SubagentTool]) -> list[list[str]]:
    """Every strongly connected component of size > 1, plus any self-loop."""
    found: list[list[str]] = []
    seen: set[str] = set()

    def walk(name: str, path: list[str]) -> None:
        if name in path:
            found.append(path[path.index(name) :])
            return
        if name in seen:
            return
        seen.add(name)
        for nxt in agents[name].spec.tools:
            if nxt in agents:
                walk(nxt, [*path, name])

    for name in agents:
        walk(name, [])
    return found


def _child_registry(
    spec: SubagentSpec, runtime: SubagentRuntime, call_id: str | None = None
) -> ToolRegistry:
    child_runtime = replace(
        runtime,
        depth=runtime.depth + 1,
        ancestors=runtime.ancestors | {spec.name},
        # A grandchild's `parent_id` is this call, which is what makes the
        # envelope a chain rather than a flat pair.
        agent_id=call_id if call_id is not None else runtime.agent_id,
    )
    allowed = ToolRegistry()
    for t in runtime.registry:
        if t.name not in spec.tools:
            continue
        if isinstance(t, SubagentTool):
            # The allowlist above is the control; this denies only the one
            # thing it cannot express — an agent already running above this
            # one, which is where a cycle would close.
            if t.spec.name in child_runtime.ancestors:
                continue
            allowed.add(t.rebound(child_runtime))
        else:
            allowed.add(t)
    return allowed


async def _drain(
    child: Agent,
    opening: str,
    spec: SubagentSpec,
    runtime: SubagentRuntime,
    call_id: str | None,
) -> AssistantMessage:
    """Run the child, relaying the events worth seeing.

    `Agent.run` is this loop with the events discarded, which is why a
    delegation used to be a black box. The envelope rides alongside the event
    rather than inside it, so `event_to_wire` maps what it always mapped.
    """
    envelope = {
        "agent": spec.name,
        "agent_id": call_id,
        "parent_id": runtime.agent_id,
        "depth": runtime.depth + 1,
    }
    last: AssistantMessage | None = None
    async for ev in child.stream(opening):
        if isinstance(ev, Done | Error):
            last = ev.message
        if runtime.on_event is not None and isinstance(ev, FORWARDED_EVENTS):
            try:
                await runtime.on_event(ev, envelope)
            except Exception as e:
                # A closed or broken relay must not take the child down with it.
                _logger.warning(
                    "subagent_event_relay_failed agent=%s call=%s error=%s",
                    spec.name,
                    call_id or "-",
                    e,
                    exc_info=e,
                )
    assert last is not None
    return last


async def _run(
    spec: SubagentSpec,
    opening: str,
    runtime: SubagentRuntime,
    call_id: str | None,
    asked_timeout: float | None = None,
) -> str:
    # Author's budget, or the caller's if they asked, clamped by the operator's
    # ceiling. Without the clamp, offering the caller a timeout would be
    # offering them none.
    budget = min(asked_timeout or spec.timeout or runtime.timeout, runtime.max_timeout)
    child = Agent(
        client=runtime.client,
        model=spec.model or runtime.model,
        tools=_child_registry(spec, runtime, call_id),
        system_prompt=spec.prompt,
        # Tool policy is inherited so a blocked call stays blocked when it is
        # delegated; everything else is dropped. See `INHERITED_EVENTS`.
        hooks=_ChildHooks(runtime.hooks) if runtime.hooks is not None else None,
    )

    # Queueing behind the concurrency limit should not leave an empty
    # transcript behind, so the file is opened only once this run is really
    # starting.
    async with runtime.limit:
        session = _open_transcript(spec, runtime, call_id)
        started = time.monotonic()
        _logger.info(
            "subagent_start agent=%s call=%s depth=%d tools=%d",
            spec.name,
            call_id or "-",
            runtime.depth,
            len(child.tools),
        )
        try:
            # Awaited inline, not spawned: the parent's interrupt path harvests
            # tool tasks rather than cancelling them, so a detached child would
            # outlive the turn that asked for it.
            final = await asyncio.wait_for(
                _drain(child, opening, spec, runtime, call_id), timeout=budget
            )
        except TimeoutError:
            _logger.warning(
                "subagent_timeout agent=%s call=%s seconds=%.0f",
                spec.name,
                call_id or "-",
                budget,
            )
            return f"[{spec.name} timed out after {budget:.0f}s]"
        finally:
            if session is not None:
                # Written even on timeout or interrupt — a partial transcript is
                # the most useful thing there is when a child misbehaves.
                session.append_many(child.history)
                session.close()

    elapsed = time.monotonic() - started
    _logger.info(
        "subagent_finished agent=%s call=%s depth=%d messages=%d tool_calls=%d "
        "seconds=%.1f transcript=%s",
        spec.name,
        call_id or "-",
        runtime.depth,
        len(child.history),
        _count_tool_calls(child.history),
        elapsed,
        session.path if session is not None else "-",
    )

    if final.stop_reason in ("error", "aborted"):
        reason = final.error_message or final.stop_reason
        return f"[{spec.name} did not finish: {reason}]"

    text = "".join(c.text for c in final.content if isinstance(c, TextContent)).strip()
    return text or f"[{spec.name} returned nothing]"


def _count_tool_calls(history: list[Any]) -> int:
    return sum(
        1
        for m in history
        if isinstance(m, AssistantMessage)
        for c in m.content
        if isinstance(c, ToolCall)
    )


def _open_transcript(
    spec: SubagentSpec, runtime: SubagentRuntime, call_id: str | None
) -> Session | None:
    """A sibling of the parent's session file, keyed on the tool call id.

    The id is what ties the child back to the exact turn that spawned it: it is
    on the parent's ToolCall and again on the ToolResultMessage that answers it.

    The parent also gets a `continued` record naming the child, so a reader can
    walk down to the delegated work instead of scanning the directory for files
    that happen to point back here.
    """
    parent_session = runtime.session
    if parent_session is None:
        return None
    parent = parent_session.path

    stem = _UNSAFE.sub("-", call_id or "").strip("-")[:64]
    base = f"{parent.stem}.{spec.name}"
    for candidate in _candidate_names(base, stem):
        path = parent.with_name(candidate + parent.suffix)
        try:
            child = Session.new(
                path,
                model=spec.model or runtime.model,
                system_prompt=spec.prompt,
                origin="subagent",
                parent_session=str(parent),
                parent_tool_call_id=call_id,
            )
        except FileExistsError:
            continue
        except OSError as e:
            _logger.warning("subagent_transcript_failed path=%s error=%s", path, e)
            return None
        # After the child opens, because the retry loop above is what decides
        # its name. A failure here costs the forward link and nothing else, so
        # the child is still returned.
        try:
            parent_session.append_continued(path=child.path.name, reason="subagent")
        except (OSError, ValueError) as e:
            _logger.warning(
                "subagent_forward_link_failed parent=%s child=%s error=%s",
                parent,
                child.path.name,
                e,
            )
        return child
    return None


def _candidate_names(base: str, stem: str) -> list[str]:
    names = [f"{base}-{stem}"] if stem else []
    # A provider that reuses or omits call ids should still get a transcript.
    names.extend(f"{base}-{i}" for i in range(100))
    return names


__all__ = [
    "SubagentRuntime",
    "SubagentSpec",
    "SubagentTool",
    "bind_subagents",
    "subagent",
    "validate",
]
