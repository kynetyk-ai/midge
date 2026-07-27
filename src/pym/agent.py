from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from pym.client import (
    Client,
    Done,
    Error,
    StreamEvent,
    StreamStart,
)
from pym.hooks import (
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
from pym.messages import (
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pym.tools import ToolRegistry


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

    async def stream(self, user_input: str | UserMessage) -> AsyncIterator[AgentEvent]:
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
                    tool_calls[i] = tc.model_copy(update={"arguments": decision.arguments})

            for tc in tool_calls:
                yield ToolExecutionStart(tool_call=tc)

            # Order must survive the partition — `zip(strict=True)` below depends on it.
            pending = [
                (i, tc)
                for i, (tc, d) in enumerate(zip(tool_calls, decisions, strict=True))
                if not (isinstance(d, ToolCallResult) and d.block)
            ]
            executed = await asyncio.gather(*(self._run_tool(tc) for _, tc in pending))

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
                            tool_call=tc, content=result.content, is_error=result.is_error
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
                yield ToolExecutionEnd(tool_call=tc, result=result)

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
        try:
            result: Any = await self.tools.invoke(tc.name, tc.arguments)
            return ToolResultMessage(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=[TextContent(text=str(result))],
                is_error=False,
            )
        except KeyError:
            return _tool_error(tc, f"Tool {tc.name!r} not found")
        except ValidationError as e:
            return _tool_error(tc, f"Invalid arguments: {e}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return _tool_error(tc, f"Tool error: {type(e).__name__}: {e}")


def _tool_error(tc: ToolCall, msg: str) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tc.id,
        tool_name=tc.name,
        content=[TextContent(text=msg)],
        is_error=True,
    )
