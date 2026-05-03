"""Tests for the Phase 5 demo notes skill.

These also serve as the *adaptability proof*: the skill loader and tool
machinery don't know anything about notes — they just discover @tool
functions in a directory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from pi.skills import load_skills

_NOTES_FILE = Path(__file__).parent.parent / "examples" / "notes_skill" / "notes.py"


@pytest.fixture
def notes_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the notes skill module fresh, with PI_NOTES_KB pointed at tmp_path."""
    monkeypatch.setenv("PI_NOTES_KB", str(tmp_path / "kb.json"))
    spec = importlib.util.spec_from_file_location(
        f"_pi_test_notes_{id(tmp_path)}", _NOTES_FILE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_add_then_read(notes_skill: ModuleType) -> None:
    out = await notes_skill.add_note.invoke(
        {"title": "Quantum gates", "content": "Hadamard, CNOT, ..."}
    )
    assert "Saved" in out
    body = await notes_skill.read_note.invoke({"title": "Quantum gates"})
    assert "Hadamard" in body
    assert "Quantum gates" in body


async def test_search_finds_match(notes_skill: ModuleType) -> None:
    await notes_skill.add_note.invoke({"title": "A", "content": "find me here"})
    await notes_skill.add_note.invoke({"title": "B", "content": "skip me"})
    out = await notes_skill.search_notes.invoke({"query": "find"})
    assert "A" in out
    assert "B" not in out


async def test_search_searches_tags(notes_skill: ModuleType) -> None:
    await notes_skill.add_note.invoke(
        {"title": "Note A", "content": "body", "tags": ["chemistry"]}
    )
    out = await notes_skill.search_notes.invoke({"query": "chem"})
    assert "Note A" in out


async def test_search_no_match(notes_skill: ModuleType) -> None:
    out = await notes_skill.search_notes.invoke({"query": "ghost"})
    assert "No notes match" in out


async def test_list_with_and_without_tag(notes_skill: ModuleType) -> None:
    await notes_skill.add_note.invoke(
        {"title": "Note A", "content": "a", "tags": ["foo"]}
    )
    await notes_skill.add_note.invoke(
        {"title": "Note B", "content": "b", "tags": ["bar"]}
    )

    everything = await notes_skill.list_notes.invoke({})
    assert "Note A" in everything
    assert "Note B" in everything

    only_foo = await notes_skill.list_notes.invoke({"tag": "foo"})
    assert "Note A" in only_foo
    assert "Note B" not in only_foo


async def test_list_empty(notes_skill: ModuleType) -> None:
    out = await notes_skill.list_notes.invoke({})
    assert "No notes" in out


async def test_duplicate_title_raises(notes_skill: ModuleType) -> None:
    await notes_skill.add_note.invoke({"title": "X", "content": "1"})
    with pytest.raises(ValueError, match="already exists"):
        await notes_skill.add_note.invoke({"title": "X", "content": "2"})


async def test_empty_slug_title_raises(notes_skill: ModuleType) -> None:
    with pytest.raises(ValueError, match="alphanumeric"):
        await notes_skill.add_note.invoke({"title": "!!!", "content": "x"})


async def test_read_missing_raises(notes_skill: ModuleType) -> None:
    with pytest.raises(KeyError):
        await notes_skill.read_note.invoke({"title": "ghost"})


async def test_read_falls_back_to_title_match(notes_skill: ModuleType) -> None:
    """Slugs are deterministic, so title 'Quantum Gates' and 'quantum gates'
    both produce the same slug. The fallback covers cases where the requested
    title differs in punctuation/whitespace beyond what slugging handles."""
    await notes_skill.add_note.invoke({"title": "Hello World", "content": "body"})
    body = await notes_skill.read_note.invoke({"title": "hello world"})
    assert "Hello World" in body


async def test_link_notes(notes_skill: ModuleType) -> None:
    await notes_skill.add_note.invoke({"title": "A", "content": "a"})
    await notes_skill.add_note.invoke({"title": "B", "content": "b"})
    out = await notes_skill.link_notes.invoke({"from_title": "A", "to_title": "B"})
    assert "Linked" in out


async def test_link_duplicate_idempotent(notes_skill: ModuleType) -> None:
    await notes_skill.add_note.invoke({"title": "A", "content": "a"})
    await notes_skill.add_note.invoke({"title": "B", "content": "b"})
    await notes_skill.link_notes.invoke({"from_title": "A", "to_title": "B"})
    out = await notes_skill.link_notes.invoke({"from_title": "A", "to_title": "B"})
    assert "already exists" in out


async def test_link_missing_raises(notes_skill: ModuleType) -> None:
    await notes_skill.add_note.invoke({"title": "A", "content": "a"})
    with pytest.raises(KeyError):
        await notes_skill.link_notes.invoke({"from_title": "A", "to_title": "ghost"})


def test_skill_loads_through_load_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of Phase 5: load_skills() picks up the notes pack
    exactly the way it picks up the built-in coding skills, with no special
    casing in core."""
    monkeypatch.setenv("PI_NOTES_KB", str(tmp_path / "kb.json"))
    registry, prompt_addition = load_skills([_NOTES_FILE.parent])
    expected = {"add_note", "search_notes", "read_note", "list_notes", "link_notes"}
    assert {t.name for t in registry} == expected
    assert "personal-notes" in prompt_addition
