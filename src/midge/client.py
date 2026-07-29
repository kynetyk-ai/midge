from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

import openai

from midge.logs import payload
from midge.messages import (
    AssistantMessage,
    Message,
    StopReason,
    TextContent,
    ToolCall,
    to_openai_messages,
)

_logger = logging.getLogger(__name__)


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


def is_retryable(exc: BaseException) -> bool:
    """Whether a failed provider call is worth another attempt.

    Deliberately short: midge targets `openai` plus OpenAI-compatible servers
    behind `base_url`, so the interesting failures are rate limits, server-side
    faults, and transport problems. Everything else — auth, bad request, model
    not found — fails the same way on every attempt.
    """
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    # APITimeoutError subclasses APIConnectionError.
    return isinstance(exc, openai.APIConnectionError)


class Client:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_attempts: int = 3,
        retry_base_delay: float = 0.5,
    ) -> None:
        self._client = openai.AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "not-needed",
            base_url=base_url,
            # The SDK's own backoff sleep ignores cancellation, so Ctrl+C during
            # a retry would do nothing. We own the sleep instead.
            max_retries=0,
        )
        self._max_attempts = max(1, max_attempts)
        self._retry_base_delay = retry_base_delay

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

        _logger.debug(
            # Every argument here is O(1) — `payload` defers the expensive part,
            # but a plain argument is evaluated whether or not DEBUG is on.
            "provider_request model=%s messages=%d tools=%d body=%s",
            model,
            len(oai_messages),
            len(oai_tools or ()),
            payload(create_kwargs),
        )

        attempt = 0
        while True:
            # Retries only happen before any content event escaped, and each
            # append is immediately followed by its start event, so a retry
            # always resumes from an empty message. `partial` itself is never
            # rebound: consumers hold the reference handed out by StreamStart.
            assert not partial.content
            text_index: int | None = None
            tool_idx_map: dict[int, int] = {}
            tool_arg_buffers: dict[int, str] = {}
            finish_reason: str | None = None

            try:
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
                                name = (
                                    tcd.function.name if tcd.function else None
                                ) or ""
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
                    except json.JSONDecodeError as e:
                        tc.arguments = {}
                        tc.arguments_error = str(e)
                        _logger.warning(
                            "tool_call_args_unparseable tool=%s id=%s error=%s buf=%.200r",
                            tc.name,
                            tc.id,
                            e,
                            buf,
                        )
                    yield ToolCallEnd(
                        content_index=content_idx,
                        tool_call=tc,
                        partial=partial,
                    )

                partial.stop_reason = _map_finish_reason(finish_reason)
                _logger.debug(
                    "provider_response model=%s finish=%s stop=%s blocks=%d attempt=%d",
                    model,
                    finish_reason,
                    partial.stop_reason,
                    len(partial.content),
                    attempt,
                )
                if attempt:
                    _logger.info("provider_recovered model=%s attempts=%d", model, attempt + 1)
                yield Done(message=partial)
                return

            except asyncio.CancelledError:
                _logger.info(
                    "provider_stream_cancelled model=%s blocks=%d", model, len(partial.content)
                )
                partial.stop_reason = "aborted"
                partial.error_message = "cancelled"
                yield Error(message=partial)
                raise
            except Exception as e:
                # Once a content event is out, the consumer has already seen
                # part of this response and a restart would duplicate it.
                committed = bool(partial.content)
                attempts_left = attempt < self._max_attempts - 1
                if not committed and attempts_left and is_retryable(e):
                    delay = self._retry_base_delay * (2**attempt)
                    attempt += 1
                    _logger.warning(
                        "provider_retry model=%s attempt=%d/%d delay=%.2f error=%s",
                        model,
                        attempt,
                        self._max_attempts,
                        delay,
                        type(e).__name__,
                    )
                    await asyncio.sleep(delay)
                    continue

                # `str(e)` is all the caller gets; without this the type and
                # traceback are gone and there is nowhere else to look.
                _logger.exception("provider_stream_failed model=%s", model)
                partial.stop_reason = "error"
                partial.error_message = f"{type(e).__name__}: {e}"
                yield Error(message=partial)
                return
