"""The OpenAI chat-completions wire format.

Registered twice, under `openai` and `openai-compatible`. One adapter rather
than two classes because ollama, vLLM, LM Studio and llama.cpp all speak this
same format — what differs is what they tolerate, and that is expressed as
`Capabilities` rather than as a second copy of the encoder.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import openai

from midge.messages import (
    AssistantMessage,
    ImageContent,
    Message,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from midge.providers.base import Capabilities, Delta, ToolCallFragment, register

_logger = logging.getLogger(__name__)

_FINISH_REASON_TO_STOP_REASON: dict[str, StopReason] = {
    "stop": "stop",
    "tool_calls": "tool_use",
    "length": "length",
    "content_filter": "error",
}


# --- request encoding -----------------------------------------------------


def _user_content(blocks: list[TextContent | ImageContent]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for b in blocks:
        if isinstance(b, TextContent):
            parts.append({"type": "text", "text": b.text})
        else:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{b.mime_type};base64,{b.data}"},
                }
            )
    return parts


def _user(m: UserMessage) -> dict[str, Any]:
    if isinstance(m.content, str):
        return {"role": "user", "content": m.content}
    return {"role": "user", "content": _user_content(m.content)}


def _assistant(m: AssistantMessage) -> dict[str, Any]:
    text_parts = [c.text for c in m.content if isinstance(c, TextContent)]
    tool_calls = [c for c in m.content if isinstance(c, ToolCall)]

    text = "".join(text_parts) if text_parts else None
    out: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in tool_calls
        ]
    return out


def _tool_result(m: ToolResultMessage) -> dict[str, Any]:
    text = "".join(c.text for c in m.content if isinstance(c, TextContent))
    if any(isinstance(c, ImageContent) for c in m.content):
        # `ToolResultResult.content` permits images, but the tool role carries
        # text only. Dropping them silently loses a hook's output with no trace.
        _logger.warning("tool_result_image_dropped tool=%s id=%s", m.tool_name, m.tool_call_id)
    return {"role": "tool", "tool_call_id": m.tool_call_id, "content": text}


def encode_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Render already-repaired history into chat-completions messages.

    Repair — dropping failed assistant turns and orphaned tool results — is
    `messages.repair_history`, because it is a fact about midge's history rather
    than about this wire format.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, UserMessage):
            out.append(_user(m))
        elif isinstance(m, AssistantMessage):
            out.append(_assistant(m))
        else:
            out.append(_tool_result(m))
    return out


def encode_body(
    *,
    messages: list[Message],
    model: str,
    tools: list[dict[str, Any]] | None,
    system: str | None,
    stream_usage: bool,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the whole request body. Free of any client, so a test can call it."""
    wire: list[dict[str, Any]] = []
    if system:
        wire.append({"role": "system", "content": system})
    wire.extend(encode_messages(messages))

    body: dict[str, Any] = {
        "model": model,
        "messages": wire,
        "stream": True,
        **kwargs,
    }
    if stream_usage:
        body.setdefault("stream_options", {"include_usage": True})
    if tools:
        body["tools"] = [{"type": "function", "function": t} for t in tools]
    return body


# --- response decoding ----------------------------------------------------


def _to_usage(raw: Any) -> Usage | None:
    """Map the usage block, tolerating servers that omit fields."""
    if raw is None:
        return None
    details = getattr(raw, "prompt_tokens_details", None)
    return Usage(
        input=getattr(raw, "prompt_tokens", 0) or 0,
        output=getattr(raw, "completion_tokens", 0) or 0,
        cached=getattr(details, "cached_tokens", 0) or 0,
    )


class OpenAIProvider:
    def __init__(
        self,
        *,
        name: str = "openai",
        capabilities: Capabilities | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.name = name
        self.capabilities = capabilities or Capabilities()
        self._client = openai.AsyncOpenAI(
            # A local server needs no credential but the SDK insists on one.
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "not-needed",
            base_url=base_url,
            # The SDK's own backoff sleep ignores cancellation, so Ctrl+C during
            # a retry would do nothing. `Client` owns the sleep instead.
            max_retries=0,
        )

    def encode(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[dict[str, Any]] | None,
        system: str | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return encode_body(
            messages=messages,
            model=model,
            tools=tools,
            system=system,
            stream_usage=self.capabilities.stream_usage,
            **kwargs,
        )

    def open(self, body: dict[str, Any]) -> Any:
        return self._client.chat.completions.create(**body)

    def decode(self, chunk: Any) -> Delta:
        # Usage rides a final chunk whose `choices` is empty, so it is read
        # before the choices are looked at rather than after.
        usage = _to_usage(getattr(chunk, "usage", None))
        if not chunk.choices:
            return Delta(usage=usage)

        choice = chunk.choices[0]
        delta = choice.delta

        fragments: list[ToolCallFragment] = []
        for tcd in delta.tool_calls or ():
            fn = tcd.function
            fragments.append(
                ToolCallFragment(
                    index=tcd.index,
                    id=tcd.id,
                    name=fn.name if fn else None,
                    arguments=(fn.arguments if fn and fn.arguments else ""),
                )
            )

        stop: StopReason | None = None
        if choice.finish_reason:
            stop = _FINISH_REASON_TO_STOP_REASON.get(choice.finish_reason, "stop")

        return Delta(
            text=delta.content or "",
            tool_calls=tuple(fragments),
            stop_reason=stop,
            usage=usage,
        )

    def is_retryable(self, exc: BaseException) -> bool:
        """Rate limits, server faults and transport problems are worth retrying.

        Everything else — auth, bad request, model not found — fails the same
        way on every attempt.
        """
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code == 429 or exc.status_code >= 500
        # APITimeoutError subclasses APIConnectionError.
        return isinstance(exc, openai.APIConnectionError)


# `capabilities=None` rather than a hardcoded default, so an operator can
# override what the adapter declares. That is how `include_usage` reaches here:
# a server that does not support `stream_options` rejects the whole turn with a
# 400 rather than ignoring the field, and there has to be a fix for that which
# does not require editing code.
register(
    "openai",
    lambda capabilities=None, **kw: OpenAIProvider(
        name="openai",
        capabilities=capabilities or Capabilities(),
        **kw,
    ),
)
register(
    "openai-compatible",
    # `stream_usage` stays on: it works on current ollama and vLLM, and the
    # override exists for the servers where it does not. Turning it off by
    # default would silently lose token counts on every local run.
    lambda capabilities=None, **kw: OpenAIProvider(
        name="openai-compatible",
        capabilities=capabilities or Capabilities(stream_usage=True),
        **kw,
    ),
)


__all__ = ["OpenAIProvider", "encode_body", "encode_messages"]
