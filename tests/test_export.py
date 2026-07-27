from __future__ import annotations

from midge.messages import (
    AssistantMessage,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from midge.session import export_html


def test_basic_text_session() -> None:
    messages: list[Message] = [
        UserMessage(content="hello"),
        AssistantMessage(
            content=[TextContent(text="hi there")],
            model="m",
            stop_reason="stop",
        ),
    ]
    out = export_html(messages, title="t", model="m")

    assert out.startswith("<!DOCTYPE html>")
    assert "<title>t</title>" in out
    assert "model: m" in out
    assert "2 messages" in out
    assert 'class="msg user"' in out
    assert 'class="msg assistant"' in out
    assert "hello" in out
    assert "hi there" in out
    assert "stop: stop" in out


def test_self_contained_no_external_assets() -> None:
    messages: list[Message] = [UserMessage(content="hi")]
    out = export_html(messages)
    assert "<link" not in out
    assert "<script" not in out
    assert "src=" not in out or 'src="data:' in out  # only data URLs allowed


def test_tool_call_renders() -> None:
    messages: list[Message] = [
        AssistantMessage(
            content=[
                TextContent(text="reading"),
                ToolCall(id="c1", name="read", arguments={"path": "/etc/hosts"}),
            ],
            stop_reason="tool_use",
        ),
    ]
    out = export_html(messages)

    assert 'class="tool-call"' in out
    assert "read" in out
    assert "c1" in out
    assert "/etc/hosts" in out


def test_tool_result_renders_in_details() -> None:
    messages: list[Message] = [
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="read",
            content=[TextContent(text="line1\nline2")],
        ),
    ]
    out = export_html(messages)

    assert "<details" in out
    assert "</details>" in out
    assert "<summary>read</summary>" in out
    assert "line1" in out
    assert "line2" in out


def test_tool_error_has_error_class_and_warning_glyph() -> None:
    messages: list[Message] = [
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="bash",
            content=[TextContent(text="exit 1")],
            is_error=True,
        ),
    ]
    out = export_html(messages)

    assert "tool-result error" in out
    assert "⚠" in out


def test_xss_user_input_escaped() -> None:
    messages: list[Message] = [
        UserMessage(content="<script>alert('xss')</script>"),
    ]
    out = export_html(messages)

    # The literal <script> must NOT survive into the output.
    assert "<script>alert" not in out
    assert "&lt;script&gt;" in out


def test_xss_tool_output_escaped() -> None:
    messages: list[Message] = [
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="bash",
            content=[TextContent(text="<img src=x onerror=alert(1)>")],
        ),
    ]
    out = export_html(messages)
    assert "<img src=x onerror" not in out
    assert "&lt;img" in out


def test_code_block_highlighted() -> None:
    messages: list[Message] = [
        AssistantMessage(
            content=[
                TextContent(
                    text="here is code:\n\n```python\ndef f():\n    return 1\n```"
                ),
            ],
        ),
    ]
    out = export_html(messages)

    assert "codehilite" in out
    # Pygments emits class-based syntax tokens for keywords like `def`
    assert 'class="k"' in out or "class='k'" in out


def test_markdown_features_render() -> None:
    messages: list[Message] = [
        UserMessage(content="**bold** and *italic* and `inline code`\n\n- a\n- b"),
    ]
    out = export_html(messages)

    assert "<strong>bold</strong>" in out
    assert "<em>italic</em>" in out
    assert "<code>inline code</code>" in out
    assert "<ul>" in out
    assert "<li>a</li>" in out


def test_empty_session() -> None:
    out = export_html([])
    assert "<!DOCTYPE html>" in out
    assert "0 messages" in out


def test_multiple_assistant_text_blocks_concatenate() -> None:
    messages: list[Message] = [
        AssistantMessage(
            content=[TextContent(text="part one"), TextContent(text="part two")],
        ),
    ]
    out = export_html(messages)
    assert "part one" in out
    assert "part two" in out


def test_user_message_with_image_renders_data_url() -> None:
    messages: list[Message] = [
        UserMessage(
            content=[
                TextContent(text="see this:"),
                ImageContent(data="aGVsbG8=", mime_type="image/png"),
            ]
        ),
    ]
    out = export_html(messages)
    assert "data:image/png;base64,aGVsbG8=" in out


def test_assistant_with_only_tool_calls_no_empty_placeholder() -> None:
    messages: list[Message] = [
        AssistantMessage(
            content=[ToolCall(id="c1", name="bash", arguments={"command": "ls"})],
            stop_reason="tool_use",
        ),
    ]
    out = export_html(messages)
    assert "(empty)" not in out
    assert "tool-call" in out


def test_default_title_when_none_given() -> None:
    out = export_html([UserMessage(content="hi")])
    assert "<title>midge session</title>" in out


def test_full_loop_session_round_trip() -> None:
    """A realistic mini-session: user → assistant tool call → tool result → assistant text."""
    messages: list[Message] = [
        UserMessage(content="list py files"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="bash", arguments={"command": "ls *.py"})],
            stop_reason="tool_use",
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="bash",
            content=[TextContent(text="a.py\nb.py")],
        ),
        AssistantMessage(
            content=[TextContent(text="There are **2 files**.")],
            stop_reason="stop",
        ),
    ]
    out = export_html(messages, title="run", model="gpt-4o")

    assert "<!DOCTYPE html>" in out and out.endswith("</html>\n")
    assert "list py files" in out
    assert "ls *.py" in out
    assert "a.py" in out and "b.py" in out
    assert "<strong>2 files</strong>" in out
    assert "4 messages" in out
