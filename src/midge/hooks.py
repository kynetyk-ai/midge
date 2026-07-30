"""Lifecycle hooks: interception points around the agent loop.

Two kinds of subscriber, following `pi-mono`'s split:

- `observe(handler)` sees every event, read-only. Return value ignored.
- `on(type, handler)` participates in that event's semantics — it can block a
  tool call, rewrite context, patch a result.

`emit()` is the only thing the loop calls. Each event type has its own
reduction rule (chain, first-cancel-or-last, early-exit-on-block); see
`Hooks._REDUCERS`.

Handlers may be sync or async. A handler that raises is logged and skipped
unless `error_mode="raise"`.

Registrations made through a *source* — today that means `load_extensions`
stamping the extension file a handler came from — can be switched off and on
by name with `set_active_sources`, which is how a profile says which hooks are
active (ADR 0001, Decision 3). Activation filters at `emit` rather than
unregistering, so a switch is a toggle rather than a reload. Handlers with no
source are always active: a profile cannot name them, so it must not be able
to disable them either.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Literal, TypeAlias, overload

from midge.messages import AssistantMessage, Message, ToolCall, UserContent, UserMessage

_logger = logging.getLogger(__name__)


# --- events ---------------------------------------------------------------
#
# Each event is a frozen dataclass with a `type` discriminant, matching the
# `AgentEvent` style in `agent.py`. Results are separate dataclasses; a
# handler returning `None` means "no opinion".


@dataclass(slots=True, frozen=True)
class SessionStart:
    path: str | None = None
    type: Literal["session_start"] = "session_start"


@dataclass(slots=True, frozen=True)
class SessionEnd:
    path: str | None = None
    type: Literal["session_end"] = "session_end"


@dataclass(slots=True, frozen=True)
class TurnStart:
    user_message: UserMessage
    system_prompt: str | None = None
    type: Literal["turn_start"] = "turn_start"


@dataclass(slots=True, frozen=True)
class Context:
    messages: list[Message] = field(default_factory=list)
    type: Literal["context"] = "context"


@dataclass(slots=True, frozen=True)
class BeforeProviderRequest:
    model: str = ""
    system: str | None = None
    tools: list[dict[str, Any]] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    type: Literal["before_provider_request"] = "before_provider_request"


@dataclass(slots=True, frozen=True)
class AfterProviderResponse:
    message: AssistantMessage
    type: Literal["after_provider_response"] = "after_provider_response"


@dataclass(slots=True, frozen=True)
class MessageEnd:
    message: AssistantMessage
    type: Literal["message_end"] = "message_end"


@dataclass(slots=True, frozen=True)
class ToolCallEvent:
    tool_call: ToolCall
    type: Literal["tool_call"] = "tool_call"


@dataclass(slots=True, frozen=True)
class ToolResultEvent:
    tool_call: ToolCall
    content: list[UserContent] = field(default_factory=list)
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass(slots=True, frozen=True)
class BeforeCompact:
    history: Sequence[Message] = ()
    cut_index: int = 0
    type: Literal["before_compact"] = "before_compact"


@dataclass(slots=True, frozen=True)
class TurnEnd:
    new_messages: list[Message] = field(default_factory=list)
    type: Literal["turn_end"] = "turn_end"


HookEvent: TypeAlias = (
    SessionStart
    | SessionEnd
    | TurnStart
    | Context
    | BeforeProviderRequest
    | AfterProviderResponse
    | MessageEnd
    | ToolCallEvent
    | ToolResultEvent
    | BeforeCompact
    | TurnEnd
)


# --- results --------------------------------------------------------------


@dataclass(slots=True)
class CancelResult:
    cancel: bool = False


@dataclass(slots=True)
class TurnStartResult:
    messages: list[Message] | None = None
    system_prompt: str | None = None


@dataclass(slots=True)
class ContextResult:
    messages: list[Message] | None = None


@dataclass(slots=True)
class ProviderRequestResult:
    model: str | None = None
    system: str | None = None
    tools: list[dict[str, Any]] | None = None
    kwargs: dict[str, Any] | None = None


@dataclass(slots=True)
class ProviderResponseResult:
    message: AssistantMessage | None = None


@dataclass(slots=True)
class ToolCallResult:
    block: bool = False
    reason: str | None = None
    arguments: dict[str, Any] | None = None


@dataclass(slots=True)
class ToolResultResult:
    content: list[UserContent] | None = None
    is_error: bool | None = None


@dataclass(slots=True)
class CompactResult:
    cancel: bool = False
    cut_index: int | None = None


Handler: TypeAlias = Callable[..., Any | Awaitable[Any]]
Unsubscribe: TypeAlias = Callable[[], None]


@dataclass(slots=True)
class _Registration:
    handler: Handler
    source: str | None
    # The extension's file stem. `source` is the full path, which is what a
    # diagnostic needs and what nobody can write in a profile — a profile names
    # `"approve"`, not `/home/…/approve.py`.
    name: str | None = None


class Hooks:
    """Registry of lifecycle handlers. Construct once, share across subsystems.

    `Hooks` is deliberately not owned by `Agent` — `before_compact` fires in
    `compaction.compact()` and the session events fire wherever the `Session`
    is owned, so the entrypoint builds this and passes it down.
    """

    def __init__(
        self,
        context: Any = None,
        *,
        error_mode: Literal["continue", "raise"] = "continue",
    ) -> None:
        self.context = context
        self.error_mode = error_mode
        self._observers: list[_Registration] = []
        self._handlers: dict[str, list[_Registration]] = {}
        self._cleanups: list[Callable[[], Any]] = []
        # None means "every source is active", which is what an entrypoint that
        # has never heard of profiles gets and stays the default forever.
        self._active: set[str] | None = None

    def set_context(self, context: Any) -> None:
        self.context = context

    def observe(
        self, handler: Handler, *, source: str | None = None, name: str | None = None
    ) -> Unsubscribe:
        reg = _Registration(handler, source, name)
        self._observers.append(reg)
        return lambda: _discard(self._observers, reg)

    def source_names(self) -> set[str]:
        """The names of every source that has registered something.

        What a profile's `hooks=(...)` is checked against — deliberately every
        registered source, not just the active ones, so naming a source that is
        currently switched off is valid rather than a validation failure.
        """
        regs = [*self._observers, *(r for rs in self._handlers.values() for r in rs)]
        return {r.name for r in regs if r.name is not None}

    @property
    def active_sources(self) -> set[str] | None:
        """The active set, or `None` for "all of them"."""
        return None if self._active is None else set(self._active)

    def set_active_sources(self, names: Iterable[str] | None) -> None:
        """Restrict which sources' handlers run. `None` restores all of them.

        **Activation, not removal.** Registrations stay in place and are
        filtered when an event is emitted. A profile switch toggles the same
        source repeatedly — off for an excursion, on again afterwards — and
        removal would mean re-importing the extension to get its handlers back,
        which is a whole reload with new `Tool` instances as a side effect.
        Filtering also keeps the operation idempotent and leaves every
        `Unsubscribe` closure valid.

        **An unnamed registration is never affected.** Only handlers registered
        through a source — which today means `load_extensions` stamping an
        extension file — can be scoped. A handler an embedder installed directly
        on this `Hooks` is not something a profile can name, so a profile must
        not be able to switch it off: an audit observer or an approval gate that
        a discovered `.py` file could silently disable would be a policy hole,
        not a feature. Same reasoning as `INHERITED_EVENTS` in `subagents.py`,
        where tool policy survives delegation.
        """
        self._active = None if names is None else set(names)

    def _is_active(self, reg: _Registration) -> bool:
        return self._active is None or reg.name is None or reg.name in self._active

    # Overloads exist so pyright checks handler signatures per event type;
    # the implementation is a single string-keyed registry.
    @overload
    def on(
        self,
        type: Literal["session_start", "session_end"],
        handler: Callable[..., CancelResult | Awaitable[CancelResult | None] | None],
        *,
        source: str | None = ...,
        name: str | None = ...,
    ) -> Unsubscribe: ...
    @overload
    def on(
        self,
        type: Literal["turn_start"],
        handler: Callable[..., TurnStartResult | Awaitable[TurnStartResult | None] | None],
        *,
        source: str | None = ...,
        name: str | None = ...,
    ) -> Unsubscribe: ...
    @overload
    def on(
        self,
        type: Literal["context"],
        handler: Callable[..., ContextResult | Awaitable[ContextResult | None] | None],
        *,
        source: str | None = ...,
        name: str | None = ...,
    ) -> Unsubscribe: ...
    @overload
    def on(
        self,
        type: Literal["before_provider_request"],
        handler: Callable[
            ..., ProviderRequestResult | Awaitable[ProviderRequestResult | None] | None
        ],
        *,
        source: str | None = ...,
        name: str | None = ...,
    ) -> Unsubscribe: ...
    @overload
    def on(
        self,
        type: Literal["after_provider_response"],
        handler: Callable[
            ..., ProviderResponseResult | Awaitable[ProviderResponseResult | None] | None
        ],
        *,
        source: str | None = ...,
        name: str | None = ...,
    ) -> Unsubscribe: ...
    @overload
    def on(
        self,
        type: Literal["tool_call"],
        handler: Callable[..., ToolCallResult | Awaitable[ToolCallResult | None] | None],
        *,
        source: str | None = ...,
        name: str | None = ...,
    ) -> Unsubscribe: ...
    @overload
    def on(
        self,
        type: Literal["tool_result"],
        handler: Callable[..., ToolResultResult | Awaitable[ToolResultResult | None] | None],
        *,
        source: str | None = ...,
        name: str | None = ...,
    ) -> Unsubscribe: ...
    @overload
    def on(
        self,
        type: Literal["before_compact"],
        handler: Callable[..., CompactResult | Awaitable[CompactResult | None] | None],
        *,
        source: str | None = ...,
        name: str | None = ...,
    ) -> Unsubscribe: ...
    @overload
    def on(
        self,
        type: Literal["message_end", "turn_end"],
        handler: Handler,
        *,
        source: str | None = ...,
        name: str | None = ...,
    ) -> Unsubscribe: ...
    @overload
    def on(
        self, type: str, handler: Handler, *, source: str | None = ..., name: str | None = ...
    ) -> Unsubscribe: ...

    def on(
        self, type: str, handler: Handler, *, source: str | None = None, name: str | None = None
    ) -> Unsubscribe:
        reg = _Registration(handler, source, name)
        self._handlers.setdefault(type, []).append(reg)
        return lambda: _discard(self._handlers.get(type, []), reg)

    def add_cleanup(self, fn: Callable[[], Any]) -> Unsubscribe:
        self._cleanups.append(fn)
        return lambda: _discard(self._cleanups, fn)

    async def clear(self) -> None:
        for fn in list(self._cleanups):
            await self._call(_Registration(fn, None))
        self._observers.clear()
        self._handlers.clear()
        self._cleanups.clear()

    async def emit(self, event: HookEvent) -> Any | None:
        # Filtering here rather than at registration is what makes activation a
        # toggle: nothing is unregistered, so switching a source back on costs a
        # set membership test instead of an extension re-import.
        for reg in list(self._observers):
            if self._is_active(reg):
                await self._call(reg, event, self.context)

        regs = [r for r in self._handlers.get(event.type, ()) if self._is_active(r)]
        if not regs:
            return None
        reducer = self._REDUCERS.get(event.type, Hooks._reduce_observation)
        return await reducer(self, event, regs)

    # --- reducers ---------------------------------------------------------
    #
    # One per event family. These encode the semantics; everything else is
    # plumbing.

    async def _reduce_observation(self, event: HookEvent, regs: list[_Registration]) -> None:
        for reg in regs:
            await self._call(reg, event, self.context)
        return None

    async def _reduce_first_cancel_or_last(
        self, event: HookEvent, regs: list[_Registration]
    ) -> Any | None:
        last: Any | None = None
        for reg in regs:
            result = await self._call(reg, event, self.context)
            if result is None:
                continue
            last = result
            if getattr(result, "cancel", False):
                return result
        return last

    async def _reduce_turn_start(
        self, event: HookEvent, regs: list[_Registration]
    ) -> TurnStartResult | None:
        assert isinstance(event, TurnStart)
        system_prompt = event.system_prompt
        injected: list[Message] = []
        for reg in regs:
            current = replace(event, system_prompt=system_prompt)
            result = await self._call(reg, current, self.context)
            if not isinstance(result, TurnStartResult):
                continue
            if result.messages:
                injected.extend(result.messages)
            if result.system_prompt is not None:
                system_prompt = result.system_prompt
        if not injected and system_prompt == event.system_prompt:
            return None
        return TurnStartResult(messages=injected or None, system_prompt=system_prompt)

    async def _reduce_context(
        self, event: HookEvent, regs: list[_Registration]
    ) -> ContextResult | None:
        assert isinstance(event, Context)
        messages = event.messages
        for reg in regs:
            result = await self._call(reg, replace(event, messages=messages), self.context)
            if isinstance(result, ContextResult) and result.messages is not None:
                messages = result.messages
        return None if messages is event.messages else ContextResult(messages=messages)

    async def _reduce_provider_request(
        self, event: HookEvent, regs: list[_Registration]
    ) -> ProviderRequestResult | None:
        assert isinstance(event, BeforeProviderRequest)
        current = event
        changed = False
        for reg in regs:
            result = await self._call(reg, current, self.context)
            if not isinstance(result, ProviderRequestResult):
                continue
            current = replace(
                current,
                model=result.model if result.model is not None else current.model,
                system=result.system if result.system is not None else current.system,
                tools=result.tools if result.tools is not None else current.tools,
                kwargs=result.kwargs if result.kwargs is not None else current.kwargs,
            )
            changed = True
        if not changed:
            return None
        return ProviderRequestResult(
            model=current.model,
            system=current.system,
            tools=current.tools,
            kwargs=current.kwargs,
        )

    async def _reduce_provider_response(
        self, event: HookEvent, regs: list[_Registration]
    ) -> ProviderResponseResult | None:
        assert isinstance(event, AfterProviderResponse)
        message = event.message
        for reg in regs:
            result = await self._call(reg, replace(event, message=message), self.context)
            if isinstance(result, ProviderResponseResult) and result.message is not None:
                message = result.message
        return None if message is event.message else ProviderResponseResult(message=message)

    async def _reduce_tool_call(
        self, event: HookEvent, regs: list[_Registration]
    ) -> ToolCallResult | None:
        assert isinstance(event, ToolCallEvent)
        current = event
        arguments: dict[str, Any] | None = None
        for reg in regs:
            result = await self._call(reg, current, self.context)
            if not isinstance(result, ToolCallResult):
                continue
            if result.arguments is not None:
                arguments = result.arguments
                # Later handlers must see the rewritten arguments.
                current = replace(
                    current, tool_call=current.tool_call.model_copy(update={"arguments": arguments})
                )
            if result.block:
                return ToolCallResult(
                    block=True, reason=result.reason, arguments=arguments
                )
        return None if arguments is None else ToolCallResult(arguments=arguments)

    async def _reduce_tool_result(
        self, event: HookEvent, regs: list[_Registration]
    ) -> ToolResultResult | None:
        assert isinstance(event, ToolResultEvent)
        current = event
        modified = False
        for reg in regs:
            result = await self._call(reg, current, self.context)
            if not isinstance(result, ToolResultResult):
                continue
            current = replace(
                current,
                content=result.content if result.content is not None else current.content,
                is_error=result.is_error if result.is_error is not None else current.is_error,
            )
            modified = True
        if not modified:
            return None
        return ToolResultResult(content=current.content, is_error=current.is_error)

    _REDUCERS: ClassVar[dict[str, Any]] = {
        "session_start": _reduce_first_cancel_or_last,
        "session_end": _reduce_first_cancel_or_last,
        "before_compact": _reduce_first_cancel_or_last,
        "turn_start": _reduce_turn_start,
        "context": _reduce_context,
        "before_provider_request": _reduce_provider_request,
        "after_provider_response": _reduce_provider_response,
        "tool_call": _reduce_tool_call,
        "tool_result": _reduce_tool_result,
        "message_end": _reduce_observation,
        "turn_end": _reduce_observation,
    }

    async def _call(self, reg: _Registration, *args: Any) -> Any | None:
        try:
            result = reg.handler(*args)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as e:
            if self.error_mode == "raise":
                raise
            # `error_mode="continue"` is the default, so this swallow is the
            # normal path — without exc_info an extension author's traceback is
            # gone for good.
            _logger.warning(
                "hook_handler_failed source=%s error=%s",
                reg.source or "-",
                type(e).__name__,
                exc_info=e,
            )
            return None


def _discard(seq: list[Any], item: Any) -> None:
    with suppress(ValueError):
        seq.remove(item)


__all__ = [
    "AfterProviderResponse",
    "BeforeCompact",
    "BeforeProviderRequest",
    "CancelResult",
    "CompactResult",
    "Context",
    "ContextResult",
    "HookEvent",
    "Hooks",
    "MessageEnd",
    "ProviderRequestResult",
    "ProviderResponseResult",
    "SessionEnd",
    "SessionStart",
    "ToolCallEvent",
    "ToolCallResult",
    "ToolResultEvent",
    "ToolResultResult",
    "TurnEnd",
    "TurnStart",
    "TurnStartResult",
]
