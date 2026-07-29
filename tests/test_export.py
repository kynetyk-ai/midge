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
from midge.persistence import CompactionRecord, Session, read_transcript
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


def test_error_message_rendered() -> None:
    messages: list[Message] = [
        AssistantMessage(
            content=[],
            stop_reason="error",
            error_message="provider returned 503",
        ),
    ]
    out = export_html(messages)

    assert "stop: error" in out
    assert "provider returned 503" in out
    assert 'class="error-message"' in out


def test_error_message_escaped() -> None:
    messages: list[Message] = [
        AssistantMessage(stop_reason="error", error_message="<script>alert(1)</script>"),
    ]
    out = export_html(messages)
    assert "<script>alert" not in out
    assert "&lt;script&gt;" in out


def test_cancelled_turn_shows_reason_alongside_partial_text() -> None:
    messages: list[Message] = [
        AssistantMessage(
            content=[TextContent(text="I was saying")],
            stop_reason="aborted",
            error_message="cancelled",
        ),
    ]
    out = export_html(messages)
    assert "I was saying" in out
    assert "cancelled" in out
    assert "(empty)" not in out


def test_no_error_message_no_error_div() -> None:
    out = export_html([AssistantMessage(content=[TextContent(text="fine")], stop_reason="stop")])
    assert 'class="error-message"' not in out


def test_tool_result_image_renders_data_url() -> None:
    messages: list[Message] = [
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="screenshot",
            content=[
                TextContent(text="captured"),
                ImageContent(data="aGVsbG8=", mime_type="image/png"),
            ],
        ),
    ]
    out = export_html(messages)
    assert "captured" in out
    assert "data:image/png;base64,aGVsbG8=" in out


def test_compaction_record_renders_as_divider() -> None:
    entries = [
        UserMessage(content="old question"),
        CompactionRecord(summary="user asked about **files**", cut_index=1),
        UserMessage(content="new question"),
    ]
    out = export_html(entries)

    assert 'class="msg compaction"' in out
    assert "1 messages summarized" in out
    assert "<strong>files</strong>" in out
    # The pre-compaction message survives in the export.
    assert "old question" in out
    assert "new question" in out
    # Compaction records are not counted as messages.
    assert "2 messages" in out


def test_export_from_session_file_keeps_compacted_messages(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m") as session:
        session.append_many(
            [
                UserMessage(content="first turn"),
                AssistantMessage(content=[TextContent(text="first answer")], stop_reason="stop"),
            ]
        )
        session.append_compaction(summary="they talked", cut_index=2)
        session.append(UserMessage(content="second turn"))

    # What the agent still holds has lost the first turn...
    assert "first turn" not in export_html(session.messages)

    # ...but the file, and therefore the export, has not.
    out = export_html(read_transcript(path)[1])
    assert "first turn" in out
    assert "first answer" in out
    assert "second turn" in out
    assert "they talked" in out


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
