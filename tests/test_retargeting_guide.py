"""`docs/retargeting.md` builds a domain from an empty directory. This builds it
from the guide's own code blocks and runs it through `cli.main`, so the guide
cannot drift from what midge actually does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from midge import cli, tui
from midge.agent import Agent

GUIDE = Path(__file__).resolve().parent.parent / "docs" / "retargeting.md"


def _blocks(lang: str) -> list[str]:
    return re.findall(rf"```{lang}\n(.*?)```", GUIDE.read_text(), re.S)


def test_the_guides_domain_runs_as_the_guide_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools, profile = _blocks("python")[:2]
    (tmp_path / "extensions").mkdir()
    (tmp_path / "extensions" / "reading.py").write_text(tools + "\n" + profile)
    (tmp_path / ".midge").mkdir()
    (tmp_path / ".midge" / "config.toml").write_text(_blocks("toml")[0])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    captured: list[Agent] = []
    monkeypatch.setattr(tui, "run_tui", lambda controls, **kw: captured.append(controls.agent))

    cli.main(["--no-session"])

    [agent] = captured
    assert {t.name for t in agent.tools} == {"add_book", "finish_book", "list_books"}
    assert agent.system_prompt is not None
    assert agent.system_prompt.startswith("You keep the user's reading list.")
    list_books = agent.tools.get("list_books")
    assert list_books is not None and list_books.read_only
