from __future__ import annotations

import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


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


class Usage(BaseModel):
    """Token counts as the provider reported them.

    No `total` — it is `input + output`, and storing a derivable number invites
    the two disagreeing. No cost, either: that needs a price table, which goes
    stale silently. Cost is a fold over the session file with whatever prices
    the reader trusts at the time they ask.
    """

    input: int = 0
    output: int = 0
    cached: int = 0


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent] = Field(default_factory=list)
    model: str = ""
    stop_reason: StopReason | None = None
    error_message: str | None = None
    usage: Usage | None = None
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


COMPACTION_PREFIX = (
    "The conversation history before this point was compacted into the "
    "following summary:\n\n<summary>\n"
)
COMPACTION_SUFFIX = "\n</summary>"


def make_summary_message(summary_text: str) -> UserMessage:
    return UserMessage(content=COMPACTION_PREFIX + summary_text + COMPACTION_SUFFIX)


def repair_history(messages: list[Message]) -> list[Message]:
    """Drop what no provider will accept, without touching the stored history.

    History is allowed to record turns that failed; a request is not. Repairing
    here rather than at the append site keeps the failure visible in the session
    log and the event stream, while making this the single boundary every request
    passes through — so damage from the loop, a hook, compaction, or a resumed
    session is normalized the same way.

    Provider-independent on purpose. Which turns are coherent is a fact about
    midge's history, not about any wire format, so every provider gets this
    rather than each re-deriving it. Encoding the survivors is the provider's
    job.
    """
    out: list[Message] = []
    issued: set[str] = set()
    for m in messages:
        if isinstance(m, UserMessage):
            out.append(m)
        elif isinstance(m, AssistantMessage):
            # A failed turn produced either tool calls that never ran or no
            # content at all. Both are rejected on the next request, and the
            # rejection outlives the session because history is persisted.
            if m.stop_reason in ("error", "aborted"):
                continue
            issued.update(c.id for c in m.content if isinstance(c, ToolCall))
            out.append(m)
        elif m.tool_call_id in issued:
            out.append(m)
    return out
