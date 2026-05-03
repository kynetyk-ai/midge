from __future__ import annotations

import json

from pi.messages import (
    AssistantMessage,
    ImageContent,
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
    [out] = to_openai_messages(
        [
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="read",
                content=[TextContent(text="file contents")],
            )
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
