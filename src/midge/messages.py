from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

_logger = logging.getLogger(__name__)


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str = ""


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    data: str
    mime_type: str


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Set when the provider's argument stream could not be parsed. Executing a
    # call whose arguments are unknown is worse than failing it: a tool whose
    # parameters all have defaults would silently run with them.
    arguments_error: str | None = None


AssistantContent = Annotated[TextContent | ToolCall, Field(discriminator="type")]
UserContent = Annotated[TextContent | ImageContent, Field(discriminator="type")]


StopReason = Literal["stop", "length", "tool_use", "error", "aborted"]


def _now_ms() -> int:
    return int(time.time() * 1000)


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[UserContent]
    timestamp: int = Field(default_factory=_now_ms)


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent] = Field(default_factory=list)
    model: str = ""
    stop_reason: StopReason | None = None
    error_message: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    timestamp: int = Field(default_factory=_now_ms)


class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str = ""
    content: list[UserContent] = Field(default_factory=list)
    is_error: bool = False
    timestamp: int = Field(default_factory=_now_ms)


Message = Annotated[
    UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="role"),
]


def _user_content_to_openai(blocks: list[TextContent | ImageContent]) -> list[dict[str, Any]]:
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


def _user_to_openai(m: UserMessage) -> dict[str, Any]:
    if isinstance(m.content, str):
        return {"role": "user", "content": m.content}
    return {"role": "user", "content": _user_content_to_openai(m.content)}


def _assistant_to_openai(m: AssistantMessage) -> dict[str, Any]:
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


def _tool_result_to_openai(m: ToolResultMessage) -> dict[str, Any]:
    text = "".join(c.text for c in m.content if isinstance(c, TextContent))
    if any(isinstance(c, ImageContent) for c in m.content):
        # `ToolResultResult.content` permits images, but the tool role carries
        # text only. Dropping them silently loses a hook's output with no trace.
        _logger.warning(
            "tool_result_image_dropped tool=%s id=%s", m.tool_name, m.tool_call_id
        )
    return {"role": "tool", "tool_call_id": m.tool_call_id, "content": text}


COMPACTION_PREFIX = (
    "The conversation history before this point was compacted into the "
    "following summary:\n\n<summary>\n"
)
COMPACTION_SUFFIX = "\n</summary>"


def make_summary_message(summary_text: str) -> UserMessage:
    return UserMessage(content=COMPACTION_PREFIX + summary_text + COMPACTION_SUFFIX)


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Render history into an OpenAI payload, dropping what a provider rejects.

    History is allowed to record turns that failed; the wire format is not.
    Repairing here rather than at the append site keeps the failure visible in
    the session log, the HTML export, and the event stream, while making this
    the single boundary every request passes through — so damage from the loop,
    a hook, compaction, or a resumed session is normalized the same way.
    """
    out: list[dict[str, Any]] = []
    issued: set[str] = set()
    for m in messages:
        if isinstance(m, UserMessage):
            out.append(_user_to_openai(m))
        elif isinstance(m, AssistantMessage):
            # A failed turn produced either tool calls that never ran or no
            # content at all. Both are rejected on the next request, and the
            # rejection outlives the session because history is persisted.
            if m.stop_reason in ("error", "aborted"):
                continue
            issued.update(c.id for c in m.content if isinstance(c, ToolCall))
            out.append(_assistant_to_openai(m))
        elif m.tool_call_id in issued:
            out.append(_tool_result_to_openai(m))
    return out
