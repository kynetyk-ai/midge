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


AgentEvent = StreamEvent | ToolExecutionStart | ToolExecutionEnd | AgentEnd


class Agent:
    def __init__(
        self,
        *,
        client: Client,
        model: str,
        tools: ToolRegistry | None = None,
        system_prompt: str | None = None,
        hooks: Hooks | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.hooks = hooks
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
                break

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

        if self.hooks is not None:
            await self.hooks.emit(TurnEnd(new_messages=new_messages))

        yield AgentEnd(new_messages=new_messages)

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
            result: Any = await self.tools.invoke(tc.name, tc.arguments)
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
