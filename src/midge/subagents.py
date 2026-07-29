"""Sub-agents: predefined nested agents, each surfaced as its own tool.

A sub-agent is a `spawn_<name>` tool. That is the whole design, and several
things fall out of it: the agent list needs no system-prompt catalogue because
tools are already self-describing; an invalid agent name is unrepresentable
because the registry enforces it; the `tool_call` hook can block one agent by
name; and recursion control is the same mechanism as tool subsetting, since the
child's registry is a subset of the parent's and the depth cap is simply
"omit the spawn tools".

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
from pathlib import Path
from typing import Any

from midge.agent import Agent
from midge.client import Client
from midge.hooks import Hooks
from midge.messages import AssistantMessage, TextContent, ToolCall
from midge.persistence import Session
from midge.tools import Tool, ToolFn, ToolRegistry, _build_params_model

DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_DEPTH = 2
DEFAULT_MAX_CONCURRENT = 4

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
    session_path: Path | None = None
    depth: int = 0
    max_depth: int = DEFAULT_MAX_DEPTH


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
        return await _run(self.spec, str(opening), self.runtime, call_id)


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
    session_path: Path | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
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
        session_path=session_path,
        depth=0,
        max_depth=max_depth,
    )
    bound = 0
    for t in registry:
        if isinstance(t, SubagentTool):
            t.runtime = runtime
            bound += 1
    if bound:
        _logger.info(
            "subagents_bound count=%d max_depth=%d max_concurrent=%d",
            bound,
            max_depth,
            max_concurrent,
        )


def _child_registry(spec: SubagentSpec, runtime: SubagentRuntime) -> ToolRegistry:
    child_runtime = replace(runtime, depth=runtime.depth + 1)
    allowed = ToolRegistry()
    for t in runtime.registry:
        if t.name not in spec.tools:
            continue
        if isinstance(t, SubagentTool):
            # The depth cap is structural: below the limit a child simply has no
            # spawn tools, so there is nothing to count and nothing to enforce.
            if child_runtime.depth + 1 > runtime.max_depth:
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
) -> str:
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
            final = await asyncio.wait_for(child.run(opening), timeout=spec.timeout)
        except TimeoutError:
            _logger.warning(
                "subagent_timeout agent=%s call=%s seconds=%.0f",
                spec.name,
                call_id or "-",
                spec.timeout,
            )
            return f"[{spec.name} timed out after {spec.timeout:.0f}s]"
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
    """
    parent = runtime.session_path
    if parent is None:
        return None

    stem = _UNSAFE.sub("-", call_id or "").strip("-")[:64]
    base = f"{parent.stem}.{spec.name}"
    for candidate in _candidate_names(base, stem):
        path = parent.with_name(candidate + parent.suffix)
        try:
            return Session.new(
                path,
                model=spec.model or runtime.model,
                system_prompt=spec.prompt,
                parent_session=str(parent),
                parent_tool_call_id=call_id,
            )
        except FileExistsError:
            continue
        except OSError as e:
            _logger.warning("subagent_transcript_failed path=%s error=%s", path, e)
            return None
    return None


def _candidate_names(base: str, stem: str) -> list[str]:
    names = [f"{base}-{stem}"] if stem else []
    # A provider that reuses or omits call ids should still get a transcript.
    names.extend(f"{base}-{i}" for i in range(100))
    return names


__all__ = [
    "DEFAULT_MAX_CONCURRENT",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_TIMEOUT",
    "SubagentRuntime",
    "SubagentSpec",
    "SubagentTool",
    "bind_subagents",
    "subagent",
]
