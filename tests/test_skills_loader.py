from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from pym.skills import BUILTIN_DIRS, load_skills
from pym.tools import Tool


def test_load_builtin_coding_skills() -> None:
    registry, prompt = load_skills(BUILTIN_DIRS)
    names = {t.name for t in registry}
    assert names == {"read", "bash", "edit", "write"}
    assert prompt == ""


def test_load_user_skill_dir(tmp_path: Path) -> None:
    skill = tmp_path / "my_skill.py"
    skill.write_text(
        "from pym.tools import tool\n\n"
        "@tool\n"
        "async def hello(name: str) -> str:\n"
        '    """Say hello."""\n'
        "    return f'hi {name}'\n"
    )
    registry, prompt_addition = load_skills([tmp_path])
    assert "hello" in registry
    assert prompt_addition == ""


def test_skips_underscore_and_init(tmp_path: Path) -> None:
    (tmp_path / "_helper.py").write_text(
        "from pym.tools import tool\n"
        "@tool\n"
        "async def hidden() -> str:\n"
        "    return 'no'\n"
    )
    (tmp_path / "__init__.py").write_text("")
    public = tmp_path / "public.py"
    public.write_text(
        "from pym.tools import tool\n"
        "@tool\n"
        "async def visible() -> str:\n"
        "    return 'ok'\n"
    )

    registry, _ = load_skills([tmp_path])
    assert "visible" in registry
    assert "hidden" not in registry
    assert len(registry) == 1


def test_system_prompt_constants_concatenated(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "from pym.tools import tool\n"
        "SYSTEM_PROMPT = 'You can do A.'\n"
        "@tool\n"
        "async def a_tool() -> str:\n"
        "    return 'a'\n"
    )
    (tmp_path / "b.py").write_text(
        "from pym.tools import tool\n"
        "SYSTEM_PROMPT = 'You can do B.'\n"
        "@tool\n"
        "async def b_tool() -> str:\n"
        "    return 'b'\n"
    )

    _, prompt = load_skills([tmp_path])
    assert "You can do A." in prompt
    assert "You can do B." in prompt


def test_empty_or_whitespace_system_prompt_ignored(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "from pym.tools import tool\n"
        "SYSTEM_PROMPT = '   '\n"
        "@tool\n"
        "async def a() -> str:\n"
        "    return 'a'\n"
    )
    _, prompt = load_skills([tmp_path])
    assert prompt == ""


def test_duplicate_tool_name_first_wins(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "a.py").write_text(
        "from pym.tools import tool\n"
        "@tool(name='shared')\n"
        "async def first() -> str:\n"
        "    return 'first'\n"
    )
    (tmp_path / "b.py").write_text(
        "from pym.tools import tool\n"
        "@tool(name='shared')\n"
        "async def second() -> str:\n"
        "    return 'second'\n"
    )

    with caplog.at_level(logging.WARNING, logger="pym.skills"):
        registry, _ = load_skills([tmp_path])

    assert len(registry) == 1
    t = registry.get("shared")
    assert t is not None
    assert isinstance(t, Tool)
    assert asyncio.run(t.invoke({})) == "first"
    assert any("shadowed" in rec.message for rec in caplog.records)


def test_missing_directory_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    nonexistent = tmp_path / "ghost"
    with caplog.at_level(logging.WARNING, logger="pym.skills"):
        registry, _ = load_skills([nonexistent])

    assert len(registry) == 0
    assert any("not found" in rec.message for rec in caplog.records)


def test_single_file_path_works(tmp_path: Path) -> None:
    f = tmp_path / "lone.py"
    f.write_text(
        "from pym.tools import tool\n"
        "@tool\n"
        "async def lone() -> str:\n"
        "    return 'ok'\n"
    )

    registry, _ = load_skills([f])
    assert "lone" in registry


def test_skill_file_can_import_sibling_module(tmp_path: Path) -> None:
    (tmp_path / "helpers.py").write_text("def shout(s):\n    return s.upper()\n")
    (tmp_path / "main.py").write_text(
        "from helpers import shout\n"
        "from pym.tools import tool\n\n"
        "@tool\n"
        "async def loud(text: str) -> str:\n"
        "    return shout(text)\n"
    )

    registry, _ = load_skills([tmp_path / "main.py"])
    assert "loud" in registry
    loud = registry.get("loud")
    assert loud is not None
    assert asyncio.run(loud.invoke({"text": "hey"})) == "HEY"


def test_failing_import_warns_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "broken.py").write_text("import this_module_does_not_exist_xyz\n")
    (tmp_path / "good.py").write_text(
        "from pym.tools import tool\n"
        "@tool\n"
        "async def good() -> str:\n"
        "    return 'ok'\n"
    )

    with caplog.at_level(logging.WARNING, logger="pym.skills"):
        registry, _ = load_skills([tmp_path])

    assert "good" in registry
    assert any("Failed to load skill" in rec.message for rec in caplog.records)


def test_two_dirs_compose(tmp_path: Path) -> None:
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    (a_dir / "tool_a.py").write_text(
        "from pym.tools import tool\n"
        "@tool\n"
        "async def alpha() -> str:\n"
        "    return 'a'\n"
    )
    (b_dir / "tool_b.py").write_text(
        "from pym.tools import tool\n"
        "@tool\n"
        "async def beta() -> str:\n"
        "    return 'b'\n"
    )

    registry, _ = load_skills([a_dir, b_dir])
    assert {t.name for t in registry} == {"alpha", "beta"}
