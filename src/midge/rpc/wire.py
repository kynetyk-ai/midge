"""Internal events to protocol frames — the one mapping layer.

Kept apart from the server so the translation stays reviewable on its own, and
so nothing can reach the wire without passing through here. `None` means an
event with no protocol counterpart: a client cannot act on it and a UI does not
render it, so it is dropped rather than given a shape nobody consumes.
"""

from __future__ import annotations

from typing import Any

from midge.agent import AgentEnd, Steered, ToolExecutionEnd, ToolExecutionStart
from midge.client import (
    Done,
    Error,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from midge.messages import TextContent, ToolCall


def event_to_wire(ev: Any) -> dict[str, Any] | None:
    if isinstance(ev, StreamStart | TextStart | TextEnd):
        return None
    if isinstance(ev, TextDelta):
        return {"type": "assistant_text_delta", "delta": ev.delta}
    if isinstance(ev, ToolCallStart):
        tc = ev.partial.content[ev.content_index]
        assert isinstance(tc, ToolCall)
        return {"type": "tool_call_start", "id": tc.id, "name": tc.name}
    if isinstance(ev, ToolCallDelta):
        tc = ev.partial.content[ev.content_index]
        assert isinstance(tc, ToolCall)
        return {"type": "tool_call_delta", "id": tc.id, "delta": ev.delta}
    if isinstance(ev, ToolCallEnd):
        return {
            "type": "tool_call_end",
            "id": ev.tool_call.id,
            "name": ev.tool_call.name,
            "arguments": ev.tool_call.arguments,
        }
    if isinstance(ev, Done):
        return {
            "type": "assistant_message_end",
            "stop_reason": ev.message.stop_reason,
            "model": ev.message.model,
        }
    if isinstance(ev, Error):
        return {
            "type": "error",
            "message": ev.message.error_message or "",
            "stop_reason": ev.message.stop_reason,
        }
    if isinstance(ev, ToolExecutionStart):
        return {
            "type": "tool_execution_start",
            "id": ev.tool_call.id,
            "name": ev.tool_call.name,
        }
    if isinstance(ev, ToolExecutionEnd):
        text = ""
        if ev.result.content and isinstance(ev.result.content[0], TextContent):
            text = ev.result.content[0].text
        return {
            "type": "tool_result",
            "tool_call_id": ev.result.tool_call_id,
            "content": text,
            "is_error": ev.result.is_error,
        }
    if isinstance(ev, Steered):
        return {
            "type": "user_message",
            "content": str(ev.message.content),
            "source": "steer",
            "queue_id": ev.queue_id,
        }
    if isinstance(ev, AgentEnd):
        return {"type": "agent_end"}
    return None
