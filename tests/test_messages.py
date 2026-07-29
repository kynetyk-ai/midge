from __future__ import annotations

import json

from midge.messages import (
    AssistantMessage,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    to_openai_messages,
)


def test_user_message_string_content_roundtrip() -> None:
    m = UserMessage(content="hello")
    loaded = UserMessage.model_validate_json(m.model_dump_json())
    assert loaded.content == "hello"
    assert loaded.role == "user"
    assert loaded.timestamp == m.timestamp


def test_user_message_multimodal_roundtrip() -> None:
    m = UserMessage(
        content=[
            TextContent(text="look at this"),
            ImageContent(data="abc==", mime_type="image/png"),
        ]
    )
    loaded = UserMessage.model_validate_json(m.model_dump_json())
    assert isinstance(loaded.content, list)
    assert len(loaded.content) == 2
    assert isinstance(loaded.content[0], TextContent)
    assert loaded.content[0].text == "look at this"
    assert isinstance(loaded.content[1], ImageContent)
    assert loaded.content[1].mime_type == "image/png"


def test_assistant_message_with_text_and_tool_calls_roundtrip() -> None:
    m = AssistantMessage(
        content=[
            TextContent(text="ok let me read it"),
            ToolCall(id="call_1", name="read", arguments={"path": "/etc/hosts"}),
        ],
        model="gpt-4o",
        stop_reason="tool_use",
    )
    loaded = AssistantMessage.model_validate_json(m.model_dump_json())
    assert loaded.model == "gpt-4o"
    assert loaded.stop_reason == "tool_use"
    assert isinstance(loaded.content[0], TextContent)
    assert isinstance(loaded.content[1], ToolCall)
    assert loaded.content[1].arguments == {"path": "/etc/hosts"}


def test_tool_result_message_roundtrip() -> None:
    m = ToolResultMessage(
        tool_call_id="call_1",
        tool_name="read",
        content=[TextContent(text="file contents here")],
        is_error=False,
    )
    loaded = ToolResultMessage.model_validate_json(m.model_dump_json())
    assert loaded.tool_call_id == "call_1"
    assert loaded.tool_name == "read"
    assert loaded.is_error is False
    assert isinstance(loaded.content[0], TextContent)
    assert loaded.content[0].text == "file contents here"


def test_to_openai_user_string() -> None:
    [out] = to_openai_messages([UserMessage(content="hi")])
    assert out == {"role": "user", "content": "hi"}


def test_to_openai_user_multimodal() -> None:
    [out] = to_openai_messages(
        [
            UserMessage(
                content=[
                    TextContent(text="see this"),
                    ImageContent(data="AAAA", mime_type="image/jpeg"),
                ]
            )
        ]
    )
    assert out["role"] == "user"
    assert out["content"] == [
        {"type": "text", "text": "see this"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
    ]


def test_to_openai_assistant_text_only() -> None:
    [out] = to_openai_messages(
        [AssistantMessage(content=[TextContent(text="hello")])]
    )
    assert out == {"role": "assistant", "content": "hello"}
    assert "tool_calls" not in out


def test_to_openai_assistant_with_tool_calls() -> None:
    [out] = to_openai_messages(
        [
            AssistantMessage(
                content=[
                    TextContent(text="reading now"),
                    ToolCall(
                        id="call_1",
                        name="read",
                        arguments={"path": "/etc/hosts", "limit": 10},
                    ),
                ]
            )
        ]
    )
    assert out["role"] == "assistant"
    assert out["content"] == "reading now"
    assert len(out["tool_calls"]) == 1
    tc = out["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "read"
    assert json.loads(tc["function"]["arguments"]) == {"path": "/etc/hosts", "limit": 10}


def test_to_openai_assistant_tool_calls_only() -> None:
    [out] = to_openai_messages(
        [
            AssistantMessage(
                content=[ToolCall(id="call_1", name="bash", arguments={"command": "ls"})]
            )
        ]
    )
    assert out["role"] == "assistant"
    assert out["content"] is None
    assert len(out["tool_calls"]) == 1


def test_to_openai_assistant_multiple_tool_calls_preserves_order() -> None:
    [out] = to_openai_messages(
        [
            AssistantMessage(
                content=[
                    ToolCall(id="call_1", name="read", arguments={"path": "a"}),
                    ToolCall(id="call_2", name="read", arguments={"path": "b"}),
                ]
            )
        ]
    )
    ids = [tc["id"] for tc in out["tool_calls"]]
    assert ids == ["call_1", "call_2"]


def test_to_openai_tool_result_text() -> None:
    _, out = to_openai_messages(
        [
            AssistantMessage(
                content=[ToolCall(id="call_1", name="read", arguments={})],
                stop_reason="tool_use",
            ),
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="read",
                content=[TextContent(text="file contents")],
            ),
        ]
    )
    assert out == {"role": "tool", "tool_call_id": "call_1", "content": "file contents"}


def test_assistant_message_can_be_mutated_in_place() -> None:
    m = AssistantMessage()
    held = m
    m.content.append(TextContent(text="hel"))
    text_block = m.content[0]
    assert isinstance(text_block, TextContent)
    text_block.text += "lo"
    m.content.append(ToolCall(id="call_1", name="read", arguments={"path": "/x"}))
    m.stop_reason = "tool_use"

    assert held is m
    assert isinstance(m.content[0], TextContent)
    assert m.content[0].text == "hello"
    assert isinstance(m.content[1], ToolCall)
    assert m.stop_reason == "tool_use"


def test_to_openai_drops_error_turn_with_orphaned_tool_call() -> None:
    """A stream that dies mid-tool-call must not poison later requests — issue #33."""
    wire = to_openai_messages(
        [
            UserMessage(content="write the file"),
            AssistantMessage(
                content=[ToolCall(id="w1", name="write", arguments={})],
                stop_reason="error",
                error_message="connection reset by peer",
            ),
            UserMessage(content="are you there?"),
        ]
    )
    assert [m["role"] for m in wire] == ["user", "user"]
    assert not any(m.get("tool_calls") for m in wire)


def test_to_openai_drops_content_less_error_turn() -> None:
    """A pre-delta failure would otherwise serialize as content=None with no tool_calls."""
    wire = to_openai_messages(
        [
            UserMessage(content="hi"),
            AssistantMessage(content=[], stop_reason="error", error_message="429"),
            UserMessage(content="retry"),
        ]
    )
    assert all(m["role"] == "user" for m in wire)
    assert not any(m.get("content") is None for m in wire)


def test_to_openai_drops_aborted_turn() -> None:
    wire = to_openai_messages(
        [
            UserMessage(content="hi"),
            AssistantMessage(content=[], stop_reason="aborted", error_message="cancelled"),
        ]
    )
    assert [m["role"] for m in wire] == ["user"]


def test_to_openai_drops_tool_result_whose_call_was_dropped() -> None:
    """Dropping the issuing assistant must not leave a tool message behind it."""
    wire = to_openai_messages(
        [
            UserMessage(content="go"),
            AssistantMessage(
                content=[ToolCall(id="x1", name="bash", arguments={})],
                stop_reason="aborted",
            ),
            ToolResultMessage(
                tool_call_id="x1",
                tool_name="bash",
                content=[TextContent(text="Interrupted")],
                is_error=True,
            ),
        ]
    )
    assert [m["role"] for m in wire] == ["user"]


def test_to_openai_keeps_successful_tool_sequences() -> None:
    wire = to_openai_messages(
        [
            UserMessage(content="go"),
            AssistantMessage(
                content=[ToolCall(id="ok1", name="read", arguments={})],
                stop_reason="tool_use",
            ),
            ToolResultMessage(
                tool_call_id="ok1",
                tool_name="read",
                content=[TextContent(text="data")],
            ),
            AssistantMessage(content=[TextContent(text="done")], stop_reason="stop"),
        ]
    )
    assert [m["role"] for m in wire] == ["user", "assistant", "tool", "assistant"]
    requested = {
        tc["id"]
        for m in wire
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    answered = {m["tool_call_id"] for m in wire if m.get("role") == "tool"}
    assert requested == answered == {"ok1"}


def test_user_message_never_splits_a_tool_group() -> None:
    """The invariant a steering drain has to respect.

    Providers reject a request where a `tool` message does not follow the
    assistant message that issued its call, and `to_openai_messages` is a single
    forward pass with no reordering — it will not repair a split. Nothing else
    in the suite checks ordering; the tool-pairing tests compare id sets.
    """
    history: list[Message] = [
        UserMessage(content="go"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="a", arguments={})],
            stop_reason="tool_use",
        ),
        ToolResultMessage(tool_call_id="c1", tool_name="a", content=[TextContent(text="ok")]),
        # Injected at the loop edge — after every result, before the next turn.
        UserMessage(content="steered"),
        AssistantMessage(content=[TextContent(text="done")], stop_reason="stop"),
    ]

    roles = [m["role"] for m in to_openai_messages(history)]

    assert roles == ["user", "assistant", "tool", "user", "assistant"]
    for i, role in enumerate(roles):
        if role == "tool":
            assert roles[i - 1] in ("assistant", "tool")


def test_a_split_tool_group_is_not_repaired() -> None:
    """Pins the failure mode, so the guarantee above is understood as something
    the drain point earns rather than something that happens to hold."""
    history: list[Message] = [
        AssistantMessage(
            content=[ToolCall(id="c1", name="a", arguments={})],
            stop_reason="tool_use",
        ),
        UserMessage(content="injected in the wrong place"),
        ToolResultMessage(tool_call_id="c1", tool_name="a", content=[TextContent(text="ok")]),
    ]

    roles = [m["role"] for m in to_openai_messages(history)]

    # The user message is passed straight through, leaving `tool` orphaned from
    # its `tool_calls`. This is what the drain point exists to avoid.
    assert roles == ["assistant", "user", "tool"]
