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
    ) -> None:
        self.client = client
        self.model = model
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.history: list[Message] = []

    async def stream(self, user_input: str | UserMessage) -> AsyncIterator[AgentEvent]:
        user_msg = (
            user_input
            if isinstance(user_input, UserMessage)
            else UserMessage(content=user_input)
        )
        self.history.append(user_msg)
        new_messages: list[Message] = [user_msg]

        while True:
            partial: AssistantMessage | None = None
            async for ev in self.client.stream(
                messages=self.history,
                model=self.model,
                tools=self.tools.schemas() or None,
                system=self.system_prompt,
            ):
                yield ev
                if isinstance(ev, StreamStart):
                    partial = ev.partial
                elif isinstance(ev, Done | Error):
                    partial = ev.message

            assert partial is not None, "stream ended without a terminal event"
            self.history.append(partial)
            new_messages.append(partial)

            if partial.stop_reason in ("error", "aborted"):
                break

            tool_calls = [c for c in partial.content if isinstance(c, ToolCall)]
            if not tool_calls:
                break

            for tc in tool_calls:
                yield ToolExecutionStart(tool_call=tc)

            results = await asyncio.gather(*(self._run_tool(tc) for tc in tool_calls))

            for tc, result in zip(tool_calls, results, strict=True):
                self.history.append(result)
                new_messages.append(result)
                yield ToolExecutionEnd(tool_call=tc, result=result)

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
