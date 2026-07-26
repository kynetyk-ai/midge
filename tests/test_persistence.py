from __future__ import annotations

import json
from pathlib import Path

import pytest

from pym.messages import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pym.persistence import VERSION, Session, SessionHeader


def test_new_session_writes_header(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="gpt-4o", system_prompt="be helpful") as s:
        assert s.header.model == "gpt-4o"
        assert s.header.system_prompt == "be helpful"
        assert s.header.version == VERSION
        assert s.messages == []

    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    head = json.loads(lines[0])
    assert head["type"] == "header"
    assert head["version"] == VERSION
    assert head["model"] == "gpt-4o"


def test_new_session_rejects_existing_path(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    p.write_text("anything")
    with pytest.raises(FileExistsError):
        Session.new(p, model="m")


def test_round_trip_user_assistant_tool_result(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m", system_prompt="x") as s:
        s.append(UserMessage(content="hi"))
        s.append(
            AssistantMessage(
                content=[
                    TextContent(text="reading"),
                    ToolCall(id="c1", name="read", arguments={"path": "x"}),
                ],
                model="m",
                stop_reason="tool_use",
            )
        )
        s.append(
            ToolResultMessage(
                tool_call_id="c1",
                tool_name="read",
                content=[TextContent(text="file contents")],
                is_error=False,
            )
        )

    with Session.load(p) as loaded:
        assert loaded.header.model == "m"
        assert loaded.header.system_prompt == "x"
        assert len(loaded.messages) == 3

        u, a, tr = loaded.messages
        assert isinstance(u, UserMessage)
        assert u.content == "hi"

        assert isinstance(a, AssistantMessage)
        assert isinstance(a.content[0], TextContent)
        assert a.content[0].text == "reading"
        assert isinstance(a.content[1], ToolCall)
        assert a.content[1].arguments == {"path": "x"}
        assert a.stop_reason == "tool_use"

        assert isinstance(tr, ToolResultMessage)
        assert tr.tool_call_id == "c1"
        assert isinstance(tr.content[0], TextContent)


def test_user_message_with_image_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        s.append(
            UserMessage(
                content=[
                    TextContent(text="see this"),
                    ImageContent(data="abc==", mime_type="image/png"),
                ]
            )
        )

    loaded = Session.load(p)
    msg = loaded.messages[0]
    assert isinstance(msg, UserMessage)
    assert isinstance(msg.content, list)
    assert isinstance(msg.content[1], ImageContent)
    assert msg.content[1].mime_type == "image/png"
    loaded.close()


def test_load_rejects_missing_header(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    p.write_text(
        '{"type":"message","data":{"role":"user","content":"hi","timestamp":1}}\n'
    )
    with pytest.raises(ValueError, match="header"):
        Session.load(p)


def test_load_rejects_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    p.write_text("")
    with pytest.raises(ValueError, match="Empty"):
        Session.load(p)


def test_load_rejects_bad_version(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    header = SessionHeader(created_at="2026-01-01T00:00:00Z", model="m").model_dump(mode="json")
    header["version"] = 999
    p.write_text(json.dumps(header) + "\n")
    with pytest.raises(ValueError, match="incompatible"):
        Session.load(p)


def test_load_skips_unknown_entry_types(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        s.append(UserMessage(content="hi"))

    with p.open("a", encoding="utf-8") as f:
        f.write('{"type":"future_thing","data":{"x":1}}\n')

    loaded = Session.load(p)
    assert len(loaded.messages) == 1
    loaded.close()


def test_compaction_entry_persisted_but_not_in_messages(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        s.append(UserMessage(content="first"))
        s.append_compaction(summary="## Goal\ntest", cut_index=1)
        s.append(UserMessage(content="second"))

    raw = p.read_text(encoding="utf-8").splitlines()
    types = [json.loads(line)["type"] for line in raw]
    assert types == ["header", "message", "compaction", "message"]

    loaded = Session.load(p)
    assert len(loaded.messages) == 2
    assert isinstance(loaded.messages[0], UserMessage)
    assert isinstance(loaded.messages[1], UserMessage)
    loaded.close()


def test_load_then_append_resumes(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        s.append(UserMessage(content="first"))

    with Session.load(p) as resumed:
        assert len(resumed.messages) == 1
        resumed.append(UserMessage(content="second"))

    final = Session.load(p)
    assert len(final.messages) == 2
    assert isinstance(final.messages[0], UserMessage)
    assert isinstance(final.messages[1], UserMessage)
    assert final.messages[0].content == "first"
    assert final.messages[1].content == "second"
    final.close()


def test_open_creates_or_resumes(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    with Session.open(p, model="m") as s:
        s.append(UserMessage(content="hi"))
        assert len(s.messages) == 1

    with Session.open(p, model="m") as s:
        assert len(s.messages) == 1
        s.append(UserMessage(content="again"))
        assert len(s.messages) == 2


def test_close_is_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    s = Session.new(p, model="m")
    s.close()
    s.close()


def test_unicode_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        s.append(UserMessage(content="héllo 🚀 中文"))

    loaded = Session.load(p)
    assert loaded.messages[0].content == "héllo 🚀 中文"
    loaded.close()
