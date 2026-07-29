from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from midge.client import (
    Client,
    Done,
    Error,
    StreamEvent,
    StreamStart,
)
from midge.hooks import (
    AfterProviderResponse,
    BeforeProviderRequest,
    Context,
    ContextResult,
    Hooks,
    MessageEnd,
    ProviderRequestResult,
    ProviderResponseResult,
    ToolCallEvent,
    ToolCallResult,
    ToolResultEvent,
    ToolResultResult,
    TurnEnd,
    TurnStart,
    TurnStartResult,
)
from midge.logs import payload
from midge.messages import (
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from midge.tools import ToolRegistry

INTERRUPTED_MESSAGE = "Interrupted by user before the tool finished."
TRUNCATED_MESSAGE = (
    "Not executed: the assistant message hit the token limit, so this tool "
    "call may be incomplete. Re-issue it if you still need it."
)
MAX_TOOL_RESULT_CHARS = 50_000

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QueuedMessage:
    id: str
    message: UserMessage


class SteeringQueue:
    """Messages a client wants injected into a run that is already going.

    Two queues with different delivery points, and the difference is the whole
    idea. **Steering** lands at a tool-call boundary inside the current run —
    "interrupt at the next safe seam", not "interrupt now": a steer issued
    during a ten-tool batch waits for that batch to finish. **Follow-up** lands
    only once the run has nothing left to do, which makes it the next turn.

    Ordering between the two is priority, not arrival: steering is drained at
    every boundary, follow-up only at quiescence, so a stream of steers delays
    a follow-up indefinitely even if it was queued first.

    Entries carry an id so a client can reconcile a delivery against what it
    queued. Matching on text, which is what pi does, is ambiguous for duplicates
    and blind to anything that is not text.

    Whatever is queued must already be a plain message. Anything whose meaning
    depends on *when* it runs — a command that invokes a handler, say — has to
    be rejected or resolved by the caller at enqueue time, not deferred to the
    boundary, so its errors surface to whoever queued it.
    """

    def __init__(self) -> None:
        self._steering: list[QueuedMessage] = []
        self._follow_up: list[QueuedMessage] = []
        self._next_id = 0

    def _wrap(self, message: str | UserMessage) -> QueuedMessage:
        self._next_id += 1
        msg = message if isinstance(message, UserMessage) else UserMessage(content=message)
        return QueuedMessage(id=f"q{self._next_id}", message=msg)

    def steer(self, message: str | UserMessage) -> str:
        entry = self._wrap(message)
        self._steering.append(entry)
        return entry.id

    def follow_up(self, message: str | UserMessage) -> str:
        entry = self._wrap(message)
        self._follow_up.append(entry)
        return entry.id

    def take_steering(self) -> list[QueuedMessage]:
        """Everything queued, not one at a time — the per-boundary throttle pi
        has exists to pace a UI that midge does not have."""
        drained, self._steering = self._steering, []
        return drained

    def take_follow_up(self) -> QueuedMessage | None:
        return self._follow_up.pop(0) if self._follow_up else None

    def clear(self) -> list[QueuedMessage]:
        """Drop everything and return it, so a caller can put it back in front
        of the user. Abort clears: pi leaves its queues alone, so aborting a
        turn silently starts a *new* run with whatever was pending, and every pi
        UI has to work around that."""
        dropped = [*self._steering, *self._follow_up]
        self._steering, self._follow_up = [], []
        return dropped

    def pending(self) -> bool:
        return bool(self._steering or self._follow_up)

    def snapshot(self) -> dict[str, list[dict[str, str]]]:
        def _render(entries: list[QueuedMessage]) -> list[dict[str, str]]:
            return [{"id": e.id, "content": str(e.message.content)} for e in entries]

        return {"steering": _render(self._steering), "follow_up": _render(self._follow_up)}


@dataclass(slots=True)
class Steered:
    """A queued message was injected into the run in progress."""

    message: UserMessage
    queue_id: str
    type: Literal["steered"] = "steered"


@dataclass(slots=True)
class ToolExecutionStart:
    tool_call: ToolCall
    type: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass(slots=True)
class ToolExecutionEnd:
    tool_call: ToolCall
    result: ToolResultMessage
    type: Literal["tool_execution_end"] = "tool_execution_end"


@dataclass(slots=True)
class AgentEnd:
    new_messages: list[Message]
    type: Literal["agent_end"] = "agent_end"


AgentEvent = StreamEvent | ToolExecutionStart | ToolExecutionEnd | Steered | AgentEnd


class Agent:
    def __init__(
        self,
        *,
        client: Client,
        model: str,
        tools: ToolRegistry | None = None,
        system_prompt: str | None = None,
        hooks: Hooks | None = None,
        steering: SteeringQueue | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.hooks = hooks
        # Owned by the entrypoint and shared, the way `hooks` and `session` are:
        # the loop only ever drains it, and whoever fills it is someone else.
        self.steering = steering
        self.history: list[Message] = []
        self._running = False

    async def stream(self, user_input: str | UserMessage) -> AsyncIterator[AgentEvent]:
        # `history` is mutated in place throughout the turn. A second concurrent
        # stream interleaves its appends with this one's, splitting tool calls
        # from their results. Callers that want to start a new turn must cancel
        # this one and await it first.
        if self._running:
            raise RuntimeError("Agent.stream is already running")
        self._running = True
        try:
            async for ev in self._stream(user_input):
                yield ev
        finally:
            self._running = False

    async def _stream(self, user_input: str | UserMessage) -> AsyncIterator[AgentEvent]:
        user_msg = (
            user_input
            if isinstance(user_input, UserMessage)
            else UserMessage(content=user_input)
        )
        system_prompt = self.system_prompt

        if self.hooks is not None:
            res = await self.hooks.emit(
                TurnStart(user_message=user_msg, system_prompt=system_prompt)
            )
            if isinstance(res, TurnStartResult):
                if res.system_prompt is not None:
                    system_prompt = res.system_prompt
                if res.messages:
                    self.history.extend(res.messages)

        self.history.append(user_msg)
        new_messages: list[Message] = [user_msg]

        # Pre-flight: anything queued while the agent was idle rides along with
        # the prompt that woke it.
        for ev in self._drain_steering(new_messages):
            yield ev

        while True:
            messages = self.history
            model = self.model
            tools = self.tools.schemas() or None
            system = system_prompt
            extra: dict[str, Any] = {}

            if self.hooks is not None:
                ctx_res = await self.hooks.emit(Context(messages=list(messages)))
                if isinstance(ctx_res, ContextResult) and ctx_res.messages is not None:
                    messages = ctx_res.messages
                req_res = await self.hooks.emit(
                    BeforeProviderRequest(
                        model=model, system=system, tools=tools, kwargs=extra
                    )
                )
                if isinstance(req_res, ProviderRequestResult):
                    model = req_res.model if req_res.model is not None else model
                    system = req_res.system if req_res.system is not None else system
                    tools = req_res.tools if req_res.tools is not None else tools
                    extra = req_res.kwargs if req_res.kwargs is not None else extra

            partial: AssistantMessage | None = None
            async for ev in self.client.stream(
                messages=messages,
                model=model,
                tools=tools,
                system=system,
                **extra,
            ):
                yield ev
                if isinstance(ev, StreamStart):
                    partial = ev.partial
                elif isinstance(ev, Done | Error):
                    partial = ev.message

            assert partial is not None, "stream ended without a terminal event"

            if self.hooks is not None:
                resp_res = await self.hooks.emit(AfterProviderResponse(message=partial))
                if isinstance(resp_res, ProviderResponseResult) and resp_res.message is not None:
                    partial = resp_res.message

            self.history.append(partial)
            new_messages.append(partial)

            if self.hooks is not None:
                await self.hooks.emit(MessageEnd(message=partial))

            if partial.stop_reason in ("error", "aborted"):
                break

            tool_calls = [c for c in partial.content if isinstance(c, ToolCall)]
            if not tool_calls:
                # A steer arriving during a turn that answered in plain text
                # re-arms the loop rather than being stranded until the next
                # prompt — which is what makes steering feel immediate.
                steered = self._drain_steering(new_messages)
                if not steered:
                    break
                for ev in steered:
                    yield ev
                continue

            # A message cut off at the token limit has tool calls the model
            # never finished emitting. Running them is a guess; fail them all
            # and let the model re-issue.
            if partial.stop_reason == "length":
                _logger.warning(
                    "tool_calls_truncated count=%d model=%s", len(tool_calls), model
                )
                for tc in tool_calls:
                    yield ToolExecutionStart(tool_call=tc)
                    result = _tool_error(tc, TRUNCATED_MESSAGE)
                    self.history.append(result)
                    new_messages.append(result)
                    yield ToolExecutionEnd(tool_call=tc, result=result)
                break

            # The assistant message with these tool calls is already in `history`.
            # Providers reject any request where a `tool_call` has no matching
            # result, so an interrupt landing anywhere below would poison every
            # later turn. `answered` tracks what we managed to append.
            answered: set[str] = set()
            tool_tasks: dict[str, asyncio.Task[ToolResultMessage]] = {}
            try:
                # `tool_call` hooks must resolve before the gather, so a blocked call
                # never executes. Decisions are gathered concurrently across calls;
                # handlers for a single call still run sequentially inside `emit`.
                if self.hooks is not None:
                    decisions = await asyncio.gather(
                        *(self.hooks.emit(ToolCallEvent(tool_call=tc)) for tc in tool_calls)
                    )
                else:
                    decisions = [None] * len(tool_calls)

                for i, (tc, decision) in enumerate(zip(tool_calls, decisions, strict=True)):
                    if isinstance(decision, ToolCallResult) and decision.arguments is not None:
                        tool_calls[i] = tc.model_copy(
                            update={"arguments": decision.arguments}
                        )

                for tc in tool_calls:
                    yield ToolExecutionStart(tool_call=tc)

                # Order must survive the partition — `zip(strict=True)` below depends on it.
                pending = [
                    (i, tc)
                    for i, (tc, d) in enumerate(zip(tool_calls, decisions, strict=True))
                    if not (isinstance(d, ToolCallResult) and d.block)
                ]
                # Kept as tasks so a cancel can still harvest whichever already
                # finished — `gather` alone would discard their results.
                for _, tc in pending:
                    tool_tasks[tc.id] = asyncio.ensure_future(self._run_tool(tc))
                executed = await asyncio.gather(*(tool_tasks[tc.id] for _, tc in pending))

                results: list[ToolResultMessage | None] = [None] * len(tool_calls)
                for (i, _), result in zip(pending, executed, strict=True):
                    results[i] = result
                for i, decision in enumerate(decisions):
                    if results[i] is None:
                        assert isinstance(decision, ToolCallResult)
                        results[i] = _tool_error(
                            tool_calls[i], decision.reason or "Blocked by hook"
                        )

                for tc, result in zip(tool_calls, results, strict=True):
                    assert result is not None
                    if self.hooks is not None:
                        patch = await self.hooks.emit(
                            ToolResultEvent(
                                tool_call=tc,
                                content=result.content,
                                is_error=result.is_error,
                            )
                        )
                        if isinstance(patch, ToolResultResult):
                            result = result.model_copy(
                                update={
                                    "content": patch.content
                                    if patch.content is not None
                                    else result.content,
                                    "is_error": patch.is_error
                                    if patch.is_error is not None
                                    else result.is_error,
                                }
                            )
                    self.history.append(result)
                    new_messages.append(result)
                    answered.add(tc.id)
                    yield ToolExecutionEnd(tool_call=tc, result=result)
            # BaseException, not Exception: `GeneratorExit` (any consumer that
            # breaks out of the `async for`), `KeyboardInterrupt`, and a hook
            # raising under `error_mode="raise"` all leave the same orphan.
            except BaseException as e:
                _logger.warning(
                    "turn_interrupted error=%s pending=%d",
                    type(e).__name__,
                    sum(1 for tc in tool_calls if tc.id not in answered),
                )
                for tc in tool_calls:
                    if tc.id in answered:
                        continue
                    task = tool_tasks.get(tc.id)
                    if task is not None and task.done() and not task.cancelled():
                        # It ran to completion before the interrupt landed; its
                        # side effects are real, so report what it actually did.
                        result = task.result() if task.exception() is None else None
                    else:
                        result = None
                    if result is None:
                        result = _tool_error(tc, INTERRUPTED_MESSAGE)
                    self.history.append(result)
                    new_messages.append(result)
                raise

            # The loop edge: every tool result for this turn is in history and
            # the next request has not been built yet. The only safe seam.
            for ev in self._drain_steering(new_messages):
                yield ev

        if self.hooks is not None:
            await self.hooks.emit(TurnEnd(new_messages=new_messages))

        yield AgentEnd(new_messages=new_messages)

    def _drain_steering(self, new_messages: list[Message]) -> list[Steered]:
        """Append anything steered into history, and say what was injected.

        Only ever called at the loop edge — after every tool result for the
        turn is appended, before the next provider request is built. Anywhere
        earlier would put a user message between an assistant message carrying
        tool calls and its results, which providers reject and which
        `to_openai_messages` does not repair.

        History and `new_messages` move together, as everywhere else: the TUI
        persists `AgentEnd.new_messages` on the normal path and `history[mark:]`
        on cancel, so a message in only one of them is lost from one of them.
        """
        if self.steering is None:
            return []
        injected: list[Steered] = []
        for entry in self.steering.take_steering():
            self.history.append(entry.message)
            new_messages.append(entry.message)
            injected.append(Steered(message=entry.message, queue_id=entry.id))
        if injected:
            _logger.info("steering_injected count=%d", len(injected))
        return injected

    async def run(self, user_input: str | UserMessage) -> AssistantMessage:
        last_assistant: AssistantMessage | None = None
        async for ev in self.stream(user_input):
            if isinstance(ev, Done | Error):
                last_assistant = ev.message
        assert last_assistant is not None
        return last_assistant

    async def _run_tool(self, tc: ToolCall) -> ToolResultMessage:
        if tc.arguments_error is not None:
            _logger.warning("tool_args_unusable tool=%s id=%s", tc.name, tc.id)
            return _tool_error(
                tc, f"Arguments could not be parsed: {tc.arguments_error}"
            )
        _logger.debug("tool_start tool=%s id=%s args=%s", tc.name, tc.id, payload(tc.arguments))
        started = time.monotonic()
        try:
            result: Any = await self.tools.invoke(tc.name, tc.arguments, call_id=tc.id)
            text = _truncate(str(result))
            _logger.info(
                "tool_ok tool=%s id=%s ms=%d chars=%d",
                tc.name,
                tc.id,
                int((time.monotonic() - started) * 1000),
                len(text),
            )
            _logger.debug("tool_result tool=%s id=%s result=%s", tc.name, tc.id, payload(text))
            return ToolResultMessage(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=[TextContent(text=text)],
                is_error=False,
            )
        except KeyError:
            _logger.warning("tool_not_found tool=%s id=%s", tc.name, tc.id)
            return _tool_error(tc, f"Tool {tc.name!r} not found")
        except ValidationError as e:
            _logger.warning("tool_args_invalid tool=%s id=%s", tc.name, tc.id)
            return _tool_error(tc, f"Invalid arguments: {e}")
        except asyncio.CancelledError:
            _logger.info("tool_cancelled tool=%s id=%s", tc.name, tc.id)
            raise
        except Exception as e:
            # exc_info because the tool body is user code: without a traceback
            # the exception type is all anyone gets, and it is rarely enough.
            _logger.warning(
                "tool_failed tool=%s id=%s error=%s",
                tc.name,
                tc.id,
                type(e).__name__,
                exc_info=e,
            )
            return _tool_error(tc, f"Tool error: {type(e).__name__}: {e}")


def _truncate(text: str) -> str:
    """Built-in tools bound their own output; extension tools have no such
    contract, and an unbounded result enters every later request."""
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    dropped = len(text) - MAX_TOOL_RESULT_CHARS
    _logger.warning("tool_result_truncated dropped=%d limit=%d", dropped, MAX_TOOL_RESULT_CHARS)
    return text[:MAX_TOOL_RESULT_CHARS] + f"\n… [truncated {dropped} characters]"


def _tool_error(tc: ToolCall, msg: str) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tc.id,
        tool_name=tc.name,
        content=[TextContent(text=msg)],
        is_error=True,
    )
