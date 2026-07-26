from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

import openai

from pym.messages import (
    AssistantMessage,
    Message,
    StopReason,
    TextContent,
    ToolCall,
    to_openai_messages,
)


@dataclass(slots=True)
class StreamStart:
    partial: AssistantMessage
    type: Literal["start"] = "start"


@dataclass(slots=True)
class TextStart:
    content_index: int
    partial: AssistantMessage
    type: Literal["text_start"] = "text_start"


@dataclass(slots=True)
class TextDelta:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True)
class TextEnd:
    content_index: int
    content: str
    partial: AssistantMessage
    type: Literal["text_end"] = "text_end"


@dataclass(slots=True)
class ToolCallStart:
    content_index: int
    partial: AssistantMessage
    type: Literal["toolcall_start"] = "toolcall_start"


@dataclass(slots=True)
class ToolCallDelta:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["toolcall_delta"] = "toolcall_delta"


@dataclass(slots=True)
class ToolCallEnd:
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage
    type: Literal["toolcall_end"] = "toolcall_end"


@dataclass(slots=True)
class Done:
    message: AssistantMessage
    type: Literal["done"] = "done"


@dataclass(slots=True)
class Error:
    message: AssistantMessage
    type: Literal["error"] = "error"


StreamEvent = (
    StreamStart
    | TextStart
    | TextDelta
    | TextEnd
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | Done
    | Error
)


_FINISH_REASON_TO_STOP_REASON: dict[str, StopReason] = {
    "stop": "stop",
    "tool_calls": "tool_use",
    "length": "length",
    "content_filter": "error",
}


def _map_finish_reason(finish_reason: str | None) -> StopReason:
    if finish_reason is None:
        return "stop"
    return _FINISH_REASON_TO_STOP_REASON.get(finish_reason, "stop")


def _tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [{"type": "function", "function": t} for t in tools]


class Client:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._client = openai.AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "not-needed",
            base_url=base_url,
        )

    async def stream(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        partial = AssistantMessage(model=model)
        yield StreamStart(partial=partial)

        text_index: int | None = None
        tool_idx_map: dict[int, int] = {}
        tool_arg_buffers: dict[int, str] = {}
        finish_reason: str | None = None

        try:
            oai_messages: list[dict[str, Any]] = []
            if system:
                oai_messages.append({"role": "system", "content": system})
            oai_messages.extend(to_openai_messages(messages))

            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": oai_messages,
                "stream": True,
                **kwargs,
            }
            oai_tools = _tools_to_openai(tools)
            if oai_tools:
                create_kwargs["tools"] = oai_tools

            stream = await self._client.chat.completions.create(**create_kwargs)

            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    if text_index is None:
                        partial.content.append(TextContent(text=""))
                        text_index = len(partial.content) - 1
                        yield TextStart(content_index=text_index, partial=partial)
                    text_block = partial.content[text_index]
                    assert isinstance(text_block, TextContent)
                    text_block.text += delta.content
                    yield TextDelta(
                        content_index=text_index,
                        delta=delta.content,
                        partial=partial,
                    )

                if delta.tool_calls:
                    for tcd in delta.tool_calls:
                        idx = tcd.index
                        if idx not in tool_idx_map:
                            tc_id = tcd.id or f"call_{idx}"
                            name = (tcd.function.name if tcd.function else None) or ""
                            tool_arg_buffers[idx] = ""
                            partial.content.append(
                                ToolCall(id=tc_id, name=name, arguments={})
                            )
                            content_idx = len(partial.content) - 1
                            tool_idx_map[idx] = content_idx
                            yield ToolCallStart(
                                content_index=content_idx, partial=partial
                            )

                        if tcd.function and tcd.function.arguments:
                            arg_chunk = tcd.function.arguments
                            tool_arg_buffers[idx] += arg_chunk
                            yield ToolCallDelta(
                                content_index=tool_idx_map[idx],
                                delta=arg_chunk,
                                partial=partial,
                            )

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            if text_index is not None:
                text_block = partial.content[text_index]
                assert isinstance(text_block, TextContent)
                yield TextEnd(
                    content_index=text_index,
                    content=text_block.text,
                    partial=partial,
                )

            for idx, content_idx in tool_idx_map.items():
                tc = partial.content[content_idx]
                assert isinstance(tc, ToolCall)
                buf = tool_arg_buffers[idx]
                try:
                    tc.arguments = json.loads(buf) if buf else {}
                except json.JSONDecodeError:
                    tc.arguments = {}
                yield ToolCallEnd(
                    content_index=content_idx,
                    tool_call=tc,
                    partial=partial,
                )

            partial.stop_reason = _map_finish_reason(finish_reason)
            yield Done(message=partial)

        except asyncio.CancelledError:
            partial.stop_reason = "aborted"
            partial.error_message = "cancelled"
            yield Error(message=partial)
            raise
        except Exception as e:
            partial.stop_reason = "error"
            partial.error_message = str(e)
            yield Error(message=partial)
