"""Sub-agents: predefined nested agents, each surfaced as its own tool.

A sub-agent is a `spawn_<name>` tool. That is the whole design, and several
things fall out of it: the agent list needs no system-prompt catalogue because
tools are already self-describing; an invalid agent name is unrepresentable
because the registry enforces it; the `tool_call` hook can block one agent by
name; and recursion control is the same mechanism as tool subsetting, since a
child only has `spawn_*` tools if its own allowlist names them.

There is deliberately **no depth cap**. The allowlist is the control, and an
author who names a `spawn_*` tool has granted nesting on purpose — a global
number would only override a declaration sitting in their file.

What must not happen is recursion with no end, and that is denied precisely: a
child never receives a `spawn_*` tool for an agent already running above it.
So `alpha -> beta` always works and `beta -> alpha` is refused only where alpha
is on the stack. Nothing is dropped and no declaration is edited, and
termination follows anyway — the ancestor set grows by one name per level from
a finite declared set, so no path outlives the number of agents. A cyclic
allowlist is a bug worth reporting, so `validate` warns about it; it does not
need to be a refusal.

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
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from midge.agent import Agent
from midge.client import Client
from midge.config import Diagnostic
from midge.hooks import Hooks
from midge.messages import AssistantMessage, TextContent, ToolCall
from midge.persistence import Session
from midge.providers import ModelRegistry
from midge.tools import Tool, ToolFn, ToolRegistry, _build_params_model

DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_CONCURRENT = 4
# The operator's ceiling. An author sets a budget per agent and a caller may ask
# for more, but nothing runs longer than this — a delegation that never returns
# is the failure worth bounding unconditionally.
DEFAULT_MAX_TIMEOUT = 900.0

_SPAWN_PREFIX = "spawn_"
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SubagentSpec:
    name: str
    prompt: str
    tools: tuple[str, ...] = ()
    model: str = ""
    timeout: float = DEFAULT_TIMEOUT


@dataclass(frozen=True, slots=True)
class SubagentRuntime:
    client: Client
    model: str
    registry: ToolRegistry
    limit: asyncio.Semaphore
    hooks: Hooks | None = None
    # The parent's open session, not just its path: a child transcript is
    # recorded in the parent, and the parent's handle is the only one that may
    # write to that file. A second handle would append past the parent's own
    # file position and its next write would overwrite the record.
    session: Session | None = None
    # Kept for the logs, where it is what makes a nested run readable. It bounds
    # nothing — `ancestors` does.
    depth: int = 0
    # The agents running above this one. A child never gets a `spawn_*` tool for
    # one of them, which is what stops recursion without touching any
    # declaration: `beta -> alpha` is denied only where it would actually
    # recurse, and works called from anywhere else.
    #
    # It is also what guarantees termination, more simply than acyclicity would:
    # the set grows by one name per level and names come from a finite declared
    # set, so no path is longer than the number of agents declared.
    ancestors: frozenset[str] = frozenset()
    max_timeout: float = DEFAULT_MAX_TIMEOUT


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
        # Opt-in: a `timeout` parameter reaches the schema only because the
        # author put it in the signature, which keeps "the signature is the tool
        # schema" true with no exception. An author who wants a fixed budget
        # simply does not offer the knob.
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
    timeout: float = DEFAULT_TIMEOUT,
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
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    max_timeout: float = DEFAULT_MAX_TIMEOUT,
) -> None:
    """Give every `SubagentTool` in `registry` what it needs to run a child.

    A no-op on a registry with no sub-agents, so entrypoints can call it
    unconditionally.
    """
    runtime = SubagentRuntime(
        client=client,
        model=model,
        registry=registry,
        limit=asyncio.Semaphore(max(1, max_concurrent)),
        hooks=hooks,
        session=session,
        depth=0,
        max_timeout=max_timeout,
    )
    bound = 0
    for t in registry:
        if isinstance(t, SubagentTool):
            t.runtime = runtime
            bound += 1
    if bound:
        _logger.info(
            "subagents_bound count=%d max_concurrent=%d max_timeout=%.0f",
            bound,
            max_concurrent,
            max_timeout,
        )


def validate(
    registry: ToolRegistry, *, models: ModelRegistry | None = None
) -> list[Diagnostic]:
    """Report what is wrong with the declared sub-agents; drop only what cannot run.

    Run once after every source is loaded, for the reason profile validation is:
    one agent's allowlist may name another declared in a different file, and no
    ordering makes per-file checking correct.

    **An unregistered model drops the agent.** A declared model no `[models]`
    entry names would otherwise surface as the vendor's 404, inside a turn, as a
    tool result. It is dropped rather than degraded, matching profiles: a tool
    the model can see but that always fails is worse than one that is absent.
    Empty registry stays permissive, as everywhere else.

    **An unknown tool name is a warning.** The agent still loads, because the
    rest of its allowlist works and a typo in one name should not cost the whole
    agent. Without this, `tools=("raed",)` silently yielded a smaller child
    registry and nothing said so.

    **A cyclic allowlist is a warning too, and drops nothing.** It is a bug in
    the declaration, and saying so is what the author needs; taking their agents
    away is not. The recursion itself cannot happen regardless — a child never
    receives a `spawn_*` tool for an agent already running above it, so the
    cycle is denied exactly where it would close and works everywhere else.
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


def _child_registry(spec: SubagentSpec, runtime: SubagentRuntime) -> ToolRegistry:
    child_runtime = replace(
        runtime,
        depth=runtime.depth + 1,
        ancestors=runtime.ancestors | {spec.name},
    )
    allowed = ToolRegistry()
    for t in runtime.registry:
        if t.name not in spec.tools:
            continue
        if isinstance(t, SubagentTool):
            # No depth cap. The allowlist above *is* the control, and an author
            # who names a `spawn_*` tool here has granted nesting on purpose —
            # a global number would only override a declaration sitting in
            # their file. The one thing that must not happen is recursion with
            # no end, and this denies exactly that: an agent already running
            # above this one, and nothing else. `beta -> alpha` still works
            # called from anywhere alpha is not already on the stack.
            if t.spec.name in child_runtime.ancestors:
                continue
            allowed.add(t.rebound(child_runtime))
        else:
            allowed.add(t)
    return allowed


async def _run(
    spec: SubagentSpec,
    opening: str,
    runtime: SubagentRuntime,
    call_id: str | None,
    asked_timeout: float | None = None,
) -> str:
    # Three layers, each owned by whoever is placed to judge it: the author sets
    # a budget for this agent, the caller may say this particular job is bigger,
    # and the operator caps the lot. The clamp is what keeps the caller's say
    # safe to offer — otherwise "specify a timeout" is "specify no timeout".
    budget = min(asked_timeout or spec.timeout, runtime.max_timeout)
    child = Agent(
        client=runtime.client,
        model=spec.model or runtime.model,
        tools=_child_registry(spec, runtime),
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
            final = await asyncio.wait_for(child.run(opening), timeout=budget)
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
    "DEFAULT_MAX_CONCURRENT",
    "DEFAULT_MAX_TIMEOUT",
    "DEFAULT_TIMEOUT",
    "SubagentRuntime",
    "SubagentSpec",
    "SubagentTool",
    "bind_subagents",
    "subagent",
    "validate",
]
