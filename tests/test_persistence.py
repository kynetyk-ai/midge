from __future__ import annotations

import json
from pathlib import Path

import pytest

from midge.messages import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from midge.persistence import (
    VERSION,
    ClearRecord,
    Session,
    SessionHeader,
    read_transcript,
)


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


def test_resume_preserves_compaction(tmp_path: Path) -> None:
    """Skipping compaction records replayed the pre-compaction history — issue #33."""
    p = tmp_path / "s.jsonl"
    s = Session.new(p, model="gpt-4o")
    for i in range(6):
        s.append(UserMessage(content=f"m{i}"))
    s.append_compaction(summary="## Goal\nstuff", cut_index=4)
    s.append(UserMessage(content="after"))
    s.close()

    loaded = Session.load(p)
    contents = [m.content for m in loaded.messages]
    assert len(loaded.messages) == 4
    assert isinstance(contents[0], str) and "<summary>" in contents[0]
    assert contents[1:] == ["m4", "m5", "after"]


def test_append_compaction_updates_in_memory_view(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    s = Session.new(p, model="gpt-4o")
    for i in range(4):
        s.append(UserMessage(content=f"m{i}"))
    s.append_compaction(summary="sum", cut_index=2)
    s.close()

    assert [m.content for m in Session.load(p).messages] == [m.content for m in s.messages]


def test_truncated_trailing_line_is_tolerated(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    s = Session.new(p, model="gpt-4o")
    s.append(UserMessage(content="kept"))
    s.close()
    with p.open("a", encoding="utf-8") as f:
        f.write('{"type":"message","data":{"role":"user","con')

    loaded = Session.load(p)
    assert [m.content for m in loaded.messages] == ["kept"]


def test_corrupt_line_that_is_not_last_still_raises(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    s = Session.new(p, model="gpt-4o")
    s.append(UserMessage(content="a"))
    s.close()
    with p.open("a", encoding="utf-8") as f:
        f.write("{not json\n")
        f.write('{"type":"message","data":{"role":"user","content":"b"}}\n')

    with pytest.raises(json.JSONDecodeError):
        Session.load(p)


def test_usage_round_trips_through_the_session_file(tmp_path: Path) -> None:
    """Pydantic carries `usage` into the JSONL with no persistence changes —
    the signal that `AssistantMessage` was the right seam for it."""
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m") as session:
        session.append(
            AssistantMessage(
                content=[TextContent(text="hi")],
                stop_reason="stop",
                usage=Usage(input=1200, output=35, cached=1024),
            )
        )

    assert '"usage"' in path.read_text(encoding="utf-8")

    restored = Session.load(path).messages[0]
    assert isinstance(restored, AssistantMessage)
    assert restored.usage is not None
    assert (restored.usage.input, restored.usage.output, restored.usage.cached) == (
        1200,
        35,
        1024,
    )


def test_sessions_without_usage_still_load(tmp_path: Path) -> None:
    # Files written before usage capture existed must keep working.
    path = tmp_path / "old.jsonl"
    path.write_text(
        '{"type":"header","version":1,"created_at":"2026-01-01","model":"m"}\n'
        '{"type":"message","data":{"role":"assistant","content":[],"stop_reason":"stop"}}\n',
        encoding="utf-8",
    )
    restored = Session.load(path).messages[0]
    assert isinstance(restored, AssistantMessage)
    assert restored.usage is None


# ---- markers: clear and session_info ----


def _messages(path: Path) -> list[str]:
    return [str(m.content) for m in Session.load(path).messages]


def test_clear_survives_a_reload(tmp_path: Path) -> None:
    """The point of the record. Without it the file replays everything and the
    clear silently undoes itself on the next resume."""
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        s.append(UserMessage(content="before"))
        s.append_clear(cut_index=1)
        s.append(UserMessage(content="after"))
        assert [str(m.content) for m in s.messages] == ["after"]

    assert _messages(p) == ["after"]


def test_a_clear_keeps_the_tail_it_was_told_to_keep(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        for i in range(4):
            s.append(UserMessage(content=f"m{i}"))
        s.append_clear(cut_index=2)

    assert _messages(p) == ["m2", "m3"]


def test_cleared_messages_are_still_in_the_file(tmp_path: Path) -> None:
    """A clear changes what a resume replays, not what happened. `export_html`
    reads the transcript, so the discarded turns stay visible there."""
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        s.append(UserMessage(content="before"))
        s.append_clear(cut_index=1)

    _, entries = read_transcript(p)
    assert any(isinstance(e, UserMessage) and e.content == "before" for e in entries)
    assert any(isinstance(e, ClearRecord) for e in entries)


def test_clear_then_compaction_summarizes_only_the_survivors(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        s.append(UserMessage(content="old"))
        s.append_clear(cut_index=1)
        s.append(UserMessage(content="kept"))
        s.append(UserMessage(content="also kept"))
        s.append_compaction(summary="S", cut_index=1)

    restored = _messages(p)
    assert any("S" in m for m in restored)
    assert not any("old" in m for m in restored)
    assert any("also kept" in m for m in restored)


def test_compaction_then_clear_leaves_nothing(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        s.append(UserMessage(content="a"))
        s.append(UserMessage(content="b"))
        s.append_compaction(summary="S", cut_index=2)
        s.append_clear(cut_index=len(s.messages))

    assert _messages(p) == []


def test_session_name_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        assert s.name is None
        s.set_name("auth refactor")
        assert s.name == "auth refactor"

    assert Session.load(p).name == "auth refactor"


def test_the_last_name_wins(tmp_path: Path) -> None:
    """A rename is another appended record, not a rewrite of the first one."""
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        s.set_name("first")
        s.append(UserMessage(content="hi"))
        s.set_name("second")

    assert Session.load(p).name == "second"


def test_a_name_does_not_become_a_message(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        s.set_name("n")
        s.append(UserMessage(content="hi"))

    assert _messages(p) == ["hi"]


def test_records_are_appended_without_rewriting_anything(tmp_path: Path) -> None:
    """The invariant the whole design exists to preserve: a rename is not a
    header rewrite. If anything before the tail changed, `read_transcript`'s
    truncated-tail recovery would no longer be sound."""
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        s.append(UserMessage(content="hi"))
        before = p.read_bytes()
        s.set_name("named")
        s.append_clear(cut_index=1)
        after = p.read_bytes()

    assert after.startswith(before), "an earlier byte of the file changed"


def test_a_truncated_marker_line_is_dropped_not_fatal(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    with Session.new(p, model="m") as s:
        s.append(UserMessage(content="hi"))
    with p.open("a", encoding="utf-8") as f:
        f.write('{"type":"clear","cut_ind')

    assert _messages(p) == ["hi"]


# ---- versioning ----


def test_an_older_session_still_loads(tmp_path: Path) -> None:
    """Bumping VERSION must not strand existing files: this build understands
    every entry type a v1 file can hold."""
    p = tmp_path / "old.jsonl"
    p.write_text(
        '{"type":"header","version":1,"created_at":"2026-01-01","model":"m"}\n'
        '{"type":"message","data":{"role":"user","content":"hi"}}\n',
        encoding="utf-8",
    )
    assert _messages(p) == ["hi"]


def test_a_newer_session_is_rejected(tmp_path: Path) -> None:
    """The asymmetry that justifies the bump. A build that would skip a `clear`
    record must decline the file rather than silently restore cleared messages."""
    p = tmp_path / "new.jsonl"
    p.write_text(
        f'{{"type":"header","version":{VERSION + 1},"created_at":"x","model":"m"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incompatible"):
        Session.load(p)
