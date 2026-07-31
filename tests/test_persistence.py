from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, TypeAlias

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
    fold_history,
    list_sessions,
    read_summary,
    read_transcript,
    resolve_session_path,
    session_chain,
    session_continuations,
    session_model,
    session_prompt,
)

Origin: TypeAlias = Literal["subagent", "profile", "fork"]


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
    """A clear changes what a resume replays, not what happened. Anything
    reading the transcript still sees the discarded turns."""
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


# --- a changed identity survives a resume (#57) ----------------------------


def test_the_model_is_the_header_until_a_record_supersedes_it(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="gpt-4o-mini") as s:
        assert s.model == "gpt-4o-mini"
        s.set_model("granite")
        assert s.model == "granite"

    assert Session.load(path).model == "granite"


def test_the_base_prompt_survives_a_resume(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m", system_prompt="original") as s:
        s.set_system_prompt("you are adversarial")

    assert Session.load(path).system_prompt == "you are adversarial"


def test_the_header_is_never_rewritten(tmp_path: Path) -> None:
    """The whole reason these are records rather than header edits.

    Truncated-tail recovery in `read_transcript` is only sound because nothing
    earlier in the file is ever rewritten, so a changed identity has to append.
    """
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="first", system_prompt="first-prompt") as s:
        s.set_model("second")
        s.set_system_prompt("second-prompt")

    loaded = Session.load(path)
    assert (loaded.header.model, loaded.header.system_prompt) == ("first", "first-prompt")
    assert (loaded.model, loaded.system_prompt) == ("second", "second-prompt")


def test_the_last_write_wins(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="a") as s:
        s.set_model("b")
        s.set_model("c")
        s.set_system_prompt("p1")
        s.set_system_prompt("p2")

    loaded = Session.load(path)
    assert (loaded.model, loaded.system_prompt) == ("c", "p2")


def test_the_records_are_appended_not_replacing_anything(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="a") as s:
        s.set_model("b")
        before = path.read_bytes()
        s.set_model("c")
        after = path.read_bytes()

    # A strict prefix: the second change added a line and touched nothing else.
    assert after.startswith(before)
    assert len(after) > len(before)


def test_an_identity_record_is_not_history(tmp_path: Path) -> None:
    # It is metadata about the agent, not a turn — `fold_history` must skip it
    # the way it skips a rename.
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m") as s:
        s.append(UserMessage(content="hello"))
        s.set_model("other")
        s.set_system_prompt("other")

    _header, entries = read_transcript(path)
    assert len(fold_history(entries)) == 1


def test_a_changed_identity_does_not_move_the_version(tmp_path: Path) -> None:
    """Deliberately not bumped, unlike `clear` in #55.

    An older build skips an entry type it does not know, which for these means
    "the change did not happen" — exactly what that build did anyway. A skipped
    `clear`, by contrast, restores messages the user discarded.
    """
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m") as s:
        s.set_model("other")
        s.set_system_prompt("other")

    header, _entries = read_transcript(path)
    assert header.version == 2


def test_never_changed_folds_to_none(tmp_path: Path) -> None:
    # None means "no record", not "no model" — the caller holds the header and
    # decides. Conflating them would make these look authoritative.
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m", system_prompt="p"):
        pass

    _header, entries = read_transcript(path)
    assert session_model(entries) is None
    assert session_prompt(entries) is None


# --- multi-file sessions: origin and forward links (#62) -------------------


def test_a_continued_record_is_not_folded_into_history(tmp_path: Path) -> None:
    """The trap in adding a record type: `fold_history` dispatches on the
    `SessionRecord` union, and its final `else` appends. A record parsed but
    left out of the union arrives in the conversation as though it were a
    message."""
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m") as s:
        s.append(UserMessage(content="hello"))
        s.append_continued(path="s.explore-c1.jsonl", reason="subagent")

    _header, entries = read_transcript(path)
    assert len(fold_history(entries)) == 1
    assert _messages(path) == ["hello"]


def test_a_continued_record_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m") as s:
        s.append_continued(path="s.explore-c1.jsonl", reason="subagent")

    _header, entries = read_transcript(path)
    (record,) = session_continuations(entries)
    assert (record.path, record.reason) == ("s.explore-c1.jsonl", "subagent")
    assert record.timestamp > 0


def test_continuations_come_back_in_the_order_they_happened(tmp_path: Path) -> None:
    # A parent has one record per child, so this is a list rather than a
    # last-write-wins read like `session_model`.
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m") as s:
        s.append_continued(path="first.jsonl", reason="subagent")
        s.append_continued(path="second.jsonl", reason="subagent")
        s.append_continued(path="third.jsonl", reason="profile")

    _header, entries = read_transcript(path)
    assert [r.path for r in session_continuations(entries)] == [
        "first.jsonl",
        "second.jsonl",
        "third.jsonl",
    ]


def test_a_forward_link_is_appended_without_rewriting_anything(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m") as s:
        s.append(UserMessage(content="hi"))
        before = path.read_bytes()
        s.append_continued(path="child.jsonl", reason="subagent")
        after = path.read_bytes()

    assert after.startswith(before), "an earlier byte of the file changed"


def test_origin_round_trips_on_the_header(tmp_path: Path) -> None:
    path = tmp_path / "child.jsonl"
    Session.new(path, model="m", origin="subagent").close()

    raw = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert raw["origin"] == "subagent"
    assert Session.load(path).header.origin == "subagent"


def test_a_root_session_has_no_origin(tmp_path: Path) -> None:
    # Absent rather than a "root" value: there is nothing to say about a file
    # that belongs to no one.
    path = tmp_path / "s.jsonl"
    Session.new(path, model="m").close()
    assert Session.load(path).header.origin is None


def test_a_header_written_before_origin_existed_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "old.jsonl"
    path.write_text(
        '{"type":"header","version":1,"created_at":"2026-01-01","model":"m"}\n',
        encoding="utf-8",
    )
    assert Session.load(path).header.origin is None


def test_an_origin_this_build_never_writes_still_loads(tmp_path: Path) -> None:
    """Why `origin` declares all three values though only `subagent` has a
    producer. VERSION deliberately does not move for this, so a newer build's
    `origin: "profile"` arrives in a file this build must read — and a Literal
    that did not list it would raise on the *header* and strand the transcript
    entirely, not merely lose the field."""
    path = tmp_path / "forked.jsonl"
    path.write_text(
        f'{{"type":"header","version":{VERSION},"created_at":"x",'
        f'"model":"m","origin":"profile"}}\n',
        encoding="utf-8",
    )
    assert Session.load(path).header.origin == "profile"


def test_forward_links_do_not_move_the_version(tmp_path: Path) -> None:
    """Same test `model_change` and `identity` passed, unlike `clear` in #55: an
    older build skipping a `continued` record is left with no forward link,
    which is what every build before this one had. Nothing is restored."""
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m", origin="subagent") as s:
        s.append_continued(path="child.jsonl", reason="subagent")

    header, _entries = read_transcript(path)
    assert header.version == 2


# --- profiles: one record for one act (#67) --------------------------------


def test_a_profile_record_is_not_folded_into_history(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m") as s:
        s.append(UserMessage(content="hello"))
        s.set_profile(name="reviewer", model="o3", system_prompt="be adversarial")

    assert _messages(path) == ["hello"]


def test_a_switch_writes_one_record_not_two(tmp_path: Path) -> None:
    """#57's constraint. A model change plus a prompt change is three facts a
    reader has to correlate and infer were one decision; naming the profile is
    what makes it one."""
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m") as s:
        s.set_profile(name="reviewer", model="o3", system_prompt="be adversarial")

    types = [json.loads(ln)["type"] for ln in path.read_text().splitlines()]
    assert types == ["header", "profile"]


def test_a_profile_sets_the_model_and_prompt_on_resume(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m", system_prompt="original") as s:
        s.set_profile(name="reviewer", model="o3", system_prompt="be adversarial")

    loaded = Session.load(path)
    assert (loaded.profile, loaded.model, loaded.system_prompt) == (
        "reviewer",
        "o3",
        "be adversarial",
    )


def test_the_last_write_wins_across_record_types(tmp_path: Path) -> None:
    """A model comes from either a `model_change` or a `profile`, and which is
    authoritative is simply which came last — not which kind it is."""
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m") as s:
        s.set_profile(name="reviewer", model="o3", system_prompt="p1")
        s.set_model("granite")
    assert Session.load(path).model == "granite"

    other = tmp_path / "t.jsonl"
    with Session.new(other, model="m") as s:
        s.set_model("granite")
        s.set_profile(name="reviewer", model="o3", system_prompt="p1")
    assert Session.load(other).model == "o3"


def test_the_last_profile_is_the_one_it_is_running_under(tmp_path: Path) -> None:
    # ADR Decision 5. A thread that has since switched away is running under
    # what it switched to, which is what excludes it as a `resume_last` target.
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m") as s:
        s.set_profile(name="builder", model="m", system_prompt="p")
        s.set_profile(name="reviewer", model="m", system_prompt="p")
    assert Session.load(path).profile == "reviewer"


def test_a_session_that_never_switched_has_no_profile(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    Session.new(path, model="m").close()
    assert Session.load(path).profile is None


def test_a_profile_record_does_not_move_the_version(tmp_path: Path) -> None:
    """An older build skipping it leaves model and prompt at whatever the
    records before it said — that build's own reading of the session. Nothing
    a user discarded is restored, which is the test `clear` failed in #55."""
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m") as s:
        s.set_profile(name="reviewer", model="o3", system_prompt="p")

    header, _entries = read_transcript(path)
    assert header.version == 2


# --- walking a session's transcripts ---------------------------------------


def _linked(
    tmp_path: Path,
    parent: str,
    child: str,
    *,
    origin: Origin,
) -> None:
    with Session.new(tmp_path / parent, model="m") as p:
        p.append_continued(path=child, reason="profile")
    Session.new(
        tmp_path / child, model="m", origin=origin, parent_session=str(tmp_path / parent)
    ).close()


def test_the_chain_is_walkable_from_any_member(tmp_path: Path) -> None:
    """Up through `parent_session` to the root, then down through `continued` —
    which is why #62 added both directions. A back-pointer alone would make
    this a directory scan, and a session owns no directory."""
    _linked(tmp_path, "root.jsonl", "root.fork.jsonl", origin="profile")

    from_root = session_chain(tmp_path / "root.jsonl")
    from_leaf = session_chain(tmp_path / "root.fork.jsonl")
    assert from_root == from_leaf == [tmp_path / "root.jsonl", tmp_path / "root.fork.jsonl"]


def test_a_lone_transcript_is_its_own_chain(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    Session.new(path, model="m").close()
    assert session_chain(path) == [path]


def test_a_deleted_link_stops_the_walk_rather_than_raising(tmp_path: Path) -> None:
    # A chain is an audit convenience; asking "where else have I been" should
    # not fail because one file was cleaned up.
    with Session.new(tmp_path / "root.jsonl", model="m") as p:
        p.append_continued(path="gone.jsonl", reason="profile")
    assert session_chain(tmp_path / "root.jsonl") == [tmp_path / "root.jsonl"]


# --- where a transcript goes -----------------------------------------------


def test_an_absolute_session_path_is_used_as_given(tmp_path: Path) -> None:
    explicit = tmp_path / "elsewhere" / "run.jsonl"
    assert resolve_session_path(explicit, directory=tmp_path / "sessions") == explicit


def test_a_relative_session_path_lands_in_the_session_directory(tmp_path: Path) -> None:
    # The behaviour change worth knowing about: `--session run.jsonl` is no
    # longer `./run.jsonl`. An absolute path is the way out.
    sessions = tmp_path / "sessions"
    assert resolve_session_path(Path("run.jsonl"), directory=sessions) == sessions / "run.jsonl"


def test_a_generated_name_is_timestamped_and_unique(tmp_path: Path) -> None:
    """The stamp so a listing sorts chronologically; the suffix because
    `Session.new` raises on collision rather than retrying, so two processes
    starting in the same second would otherwise fail at startup."""
    sessions = tmp_path / "sessions"
    first = resolve_session_path(None, directory=sessions)
    second = resolve_session_path(None, directory=sessions)
    assert first is not None and second is not None
    assert first.parent == sessions
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}\.jsonl", first.name)
    assert first != second


def test_disabled_means_no_transcript(tmp_path: Path) -> None:
    assert resolve_session_path(None, directory=tmp_path, enabled=False) is None


def test_a_named_path_outranks_a_default_off(tmp_path: Path) -> None:
    # Naming a file is a deliberate request; `enabled=False` is only a default.
    named = resolve_session_path(Path("run.jsonl"), directory=tmp_path, enabled=False)
    assert named == tmp_path / "run.jsonl"


def test_a_multiline_prompt_round_trips(tmp_path: Path) -> None:
    # `set_session_name` flattens newlines because a name is a display string;
    # a system prompt is not, and JSON escaping keeps it on one line anyway.
    prompt = "You review adversarially.\n\nRules:\n- assume it is wrong\n"
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m") as s:
        s.set_system_prompt(prompt)

    assert Session.load(path).system_prompt == prompt
    assert len([ln for ln in path.read_text().splitlines() if ln.strip()]) == 2


# --- session discovery ----------------------------------------------------


def _session(dir: Path, stem: str, *, messages: int = 0, name: str | None = None,
             origin: Origin | None = None, model: str = "m") -> Path:
    path = dir / f"{stem}.jsonl"
    with Session.new(path, model=model, origin=origin) as s:
        for i in range(messages):
            s.append(UserMessage(content=f"m{i}"))
        if name is not None:
            s.set_name(name)
    return path


def test_a_summary_reports_what_a_picker_shows(tmp_path: Path) -> None:
    path = _session(tmp_path, "a", messages=3, name="auth refactor", model="gpt-4o")

    summary = read_summary(path)

    assert summary is not None
    assert summary.name == "auth refactor"
    assert summary.model == "gpt-4o"
    assert summary.messages == 3
    assert summary.origin is None


def test_a_summary_takes_the_last_name(tmp_path: Path) -> None:
    """The cheap reader has its own scan, so `Session.load` agreeing proves
    nothing about it."""
    path = tmp_path / "a.jsonl"
    with Session.new(path, model="m") as s:
        s.set_name("first")
        s.set_name("second")

    summary = read_summary(path)
    assert summary is not None and summary.name == "second"


def test_an_unnamed_session_has_no_name(tmp_path: Path) -> None:
    summary = read_summary(_session(tmp_path, "a"))
    assert summary is not None and summary.name is None


def test_a_summary_does_not_parse_the_messages(tmp_path: Path) -> None:
    """The count is a line-type test, so a message this build cannot validate
    still counts. That is the point: a listing must not be the thing that fails
    on a transcript written by a newer build."""
    path = _session(tmp_path, "a", messages=1)
    with path.open("a", encoding="utf-8") as f:
        f.write('{"type": "message", "data": {"role": "from_the_future"}}\n')

    summary = read_summary(path)
    assert summary is not None and summary.messages == 2


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("empty", ""),
        ("blank", "\n\n"),
        ("headerless", '{"type": "message", "data": {}}\n'),
        ("not json", "hello\n"),
        ("newer version", '{"type":"header","version":999,"created_at":"x","model":"m"}\n'),
    ],
)
def test_an_unreadable_file_is_skipped_not_raised(tmp_path: Path, label: str, content: str) -> None:
    # `read_transcript` raises on all of these, which is right when someone asked
    # for *that* transcript. In a listing one bad file must not hide the rest.
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")
    assert read_summary(path) is None


def test_a_truncated_final_line_still_summarizes(tmp_path: Path) -> None:
    # `append` is write-then-flush, not atomic, so a crash mid-write leaves a
    # partial last line — the same case `read_transcript` recovers from.
    path = _session(tmp_path, "a", messages=2)
    with path.open("a", encoding="utf-8") as f:
        f.write('{"type": "session_i')

    summary = read_summary(path)
    assert summary is not None and summary.messages == 2


def test_a_listing_is_newest_first(tmp_path: Path) -> None:
    # Ordered by the header's `created_at`, not the filename: the stamp in a
    # generated name agrees with it, but the header is the fact and a
    # `--session` path names whatever the user liked.
    _session(tmp_path, "zzz", name="older")
    _session(tmp_path, "aaa", name="middle")
    _session(tmp_path, "mmm", name="newest")

    assert [s.name for s in list_sessions(tmp_path)] == ["newest", "middle", "older"]


def test_delegations_and_excursions_are_not_conversations(tmp_path: Path) -> None:
    # They are siblings on disk because a child writes beside its parent, but
    # reopening one would resume the middle of a tool call.
    _session(tmp_path, "root", name="root")
    _session(tmp_path, "child", origin="subagent")
    _session(tmp_path, "excursion", origin="profile")

    assert [s.name for s in list_sessions(tmp_path)] == ["root"]
    assert len(list_sessions(tmp_path, roots_only=False)) == 3


def test_one_bad_file_does_not_hide_the_others(tmp_path: Path) -> None:
    _session(tmp_path, "good", name="fine")
    (tmp_path / "bad").write_text("garbage\n", encoding="utf-8")

    assert [s.name for s in list_sessions(tmp_path)] == ["fine"]


def test_a_missing_directory_is_an_empty_listing(tmp_path: Path) -> None:
    # Nothing has been recorded yet, which is not an error.
    assert list_sessions(tmp_path / "nope") == []
