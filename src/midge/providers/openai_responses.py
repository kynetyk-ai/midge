"""The OpenAI Responses wire format, registered as `openai`.

OpenAI's reasoning models accept function tools on chat completions only with
reasoning switched off, so OpenAI itself is spoken to here. Chat completions
stays as `openai-compatible`, for the servers that implement nothing else.

Requests go out with `store: false`, because midge resends the whole history
every turn and nothing needs to live on OpenAI's side. The model's reasoning
then has to travel with the history instead: each response's output items,
encrypted reasoning included, are kept on the assistant message under `extra`
and replayed verbatim on the next request to the same model. The API rejects a
reasoning item replayed without the item that followed it, which is why the
whole output is kept rather than the reasoning alone.

Everything that is true of OpenAI rather than of a wire format — retry
classification, `Retry-After`, the rate-limit cool-off, what a 401 means — is
inherited from the chat-completions adapter.
"""

from __future__ import annotations

import json
from typing import Any

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
from midge.providers.openai_compat import OpenAIProvider

OUTPUT_KEY = "openai_responses_output"

# Failures reported inside a stream rather than as an HTTP status, which are
# worth another attempt for the same reasons a 429 or a 5xx is.
_RETRYABLE_CODES = frozenset({"server_error", "rate_limit_exceeded"})


class ResponseFailed(Exception):
    """The stream ended with `response.failed` or an `error` event."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


# --- request encoding -----------------------------------------------------


def _parts(blocks: list[TextContent | ImageContent]) -> list[dict[str, Any]]:
    return [
        {"type": "input_text", "text": b.text}
        if isinstance(b, TextContent)
        else {"type": "input_image", "image_url": f"data:{b.mime_type};base64,{b.data}"}
        for b in blocks
    ]


def _assistant(m: AssistantMessage, model: str) -> list[dict[str, Any]]:
    # Encrypted reasoning is only readable by the model that wrote it.
    stored = m.extra.get(OUTPUT_KEY)
    if stored and m.model == model:
        return list(stored)
    items: list[dict[str, Any]] = []
    text = "".join(c.text for c in m.content if isinstance(c, TextContent))
    if text:
        items.append({"role": "assistant", "content": text})
    items.extend(
        {
            "type": "function_call",
            "call_id": c.id,
            "name": c.name,
            "arguments": json.dumps(c.arguments),
        }
        for c in m.content
        if isinstance(c, ToolCall)
    )
    return items


def _tool_result(m: ToolResultMessage) -> dict[str, Any]:
    output: str | list[dict[str, Any]]
    if any(isinstance(c, ImageContent) for c in m.content):
        output = _parts(m.content)
    else:
        output = "".join(c.text for c in m.content if isinstance(c, TextContent))
    return {"type": "function_call_output", "call_id": m.tool_call_id, "output": output}


def encode_input(messages: list[Message], model: str) -> list[dict[str, Any]]:
    """Render already-repaired history into Responses input items."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, UserMessage):
            content = m.content if isinstance(m.content, str) else _parts(m.content)
            out.append({"role": "user", "content": content})
        elif isinstance(m, AssistantMessage):
            out.extend(_assistant(m, model))
        else:
            out.append(_tool_result(m))
    return out


def encode_body(
    *,
    messages: list[Message],
    model: str,
    tools: list[dict[str, Any]] | None,
    system: str | None,
    **kwargs: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": encode_input(messages, model),
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        **kwargs,
    }
    if system:
        body["instructions"] = system
    if tools:
        # Strict mode requires every property to be required, which a tool
        # parameter with a default is not. The API turns it on unless told.
        body["tools"] = [{"type": "function", **t, "strict": False} for t in tools]
    return body


# --- response decoding ----------------------------------------------------


def _usage(raw: Any) -> Usage | None:
    if raw is None:
        return None
    details = getattr(raw, "input_tokens_details", None)
    return Usage(
        input=raw.input_tokens or 0,
        output=raw.output_tokens or 0,
        cached=getattr(details, "cached_tokens", 0) or 0,
    )


def _stop_reason(response: Any) -> StopReason:
    if response.status == "incomplete":
        reason = getattr(response.incomplete_details, "reason", None)
        return "length" if reason == "max_output_tokens" else "error"
    if any(item.type == "function_call" for item in response.output):
        return "tool_use"
    return "stop"


def _finished(response: Any) -> Delta:
    return Delta(
        stop_reason=_stop_reason(response),
        usage=_usage(response.usage),
        extra={OUTPUT_KEY: [item.model_dump(exclude_none=True) for item in response.output]},
    )


class OpenAIResponsesProvider(OpenAIProvider):
    def encode(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[dict[str, Any]] | None,
        system: str | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return encode_body(messages=messages, model=model, tools=tools, system=system, **kwargs)

    def open(self, body: dict[str, Any]) -> Any:
        return self._client.responses.create(**body)

    def decode(self, chunk: Any) -> Delta:
        # `output_index` is stable per item for the whole response, so it
        # groups a call's argument deltas the way `ToolCallFragment.index` needs.
        # The `added` event's own `arguments` is ignored: the deltas carry them.
        kind = chunk.type
        if kind == "response.output_text.delta":
            return Delta(text=chunk.delta)
        if kind == "response.output_item.added" and chunk.item.type == "function_call":
            fragment = ToolCallFragment(
                index=chunk.output_index, id=chunk.item.call_id, name=chunk.item.name
            )
            return Delta(tool_calls=(fragment,))
        if kind == "response.function_call_arguments.delta":
            fragment = ToolCallFragment(index=chunk.output_index, arguments=chunk.delta)
            return Delta(tool_calls=(fragment,))
        if kind in ("response.completed", "response.incomplete"):
            return _finished(chunk.response)
        if kind == "response.failed":
            error = chunk.response.error
            raise ResponseFailed(
                getattr(error, "message", None) or "response failed",
                getattr(error, "code", None),
            )
        if kind == "error":
            raise ResponseFailed(chunk.message, chunk.code)
        return Delta()

    def is_retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, ResponseFailed):
            return exc.code in _RETRYABLE_CODES
        return super().is_retryable(exc)


register(
    "openai",
    lambda capabilities=None, **kw: OpenAIResponsesProvider(
        name="openai",
        capabilities=capabilities or Capabilities(),
        **kw,
    ),
)


__all__ = ["OUTPUT_KEY", "OpenAIResponsesProvider", "ResponseFailed", "encode_body"]
