from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from midge import cli
from midge.agent import Agent
from midge.client import Client
from midge.messages import UserMessage
from midge.skills import (
    Skill,
    default_skill_dirs,
    find_skill,
    load_skills,
    skill_message,
    skills_prompt,
)
from tests.fakes import finish, install, say

VALID = "Deploys the service and runs the release checklist."


def write_skill(
    directory: Path,
    *,
    frontmatter: str = f"name: deploy\ndescription: {VALID}",
    body: str = "# Deploy\n\nRun `./scripts/go.sh`.",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return path


def test_loads_skill_from_directory(tmp_path: Path) -> None:
    write_skill(tmp_path / "deploy")
    (skill,) = load_skills([tmp_path])

    assert skill.name == "deploy"
    assert skill.description == VALID
    assert skill.path == (tmp_path / "deploy" / "SKILL.md").resolve()
    assert skill.base_dir == (tmp_path / "deploy").resolve()
    assert skill.model_invocable


def test_name_falls_back_to_directory_name(tmp_path: Path) -> None:
    write_skill(tmp_path / "release-notes", frontmatter=f"description: {VALID}")
    (skill,) = load_skills([tmp_path])
    assert skill.name == "release-notes"


def test_frontmatter_name_may_differ_from_directory(tmp_path: Path) -> None:
    # The spec requires these to match; midge does not, so shared skill
    # directories from other harnesses load cleanly.
    write_skill(tmp_path / "some-folder", frontmatter=f"name: deploy\ndescription: {VALID}")
    (skill,) = load_skills([tmp_path])
    assert skill.name == "deploy"


def test_skill_directory_is_a_leaf(tmp_path: Path) -> None:
    write_skill(tmp_path / "outer")
    write_skill(tmp_path / "outer" / "references", frontmatter=f"description: {VALID}")

    skills = load_skills([tmp_path])
    assert [s.name for s in skills] == ["deploy"]


def test_finds_nested_skill_directories(tmp_path: Path) -> None:
    write_skill(tmp_path / "a" / "one", frontmatter=f"name: one\ndescription: {VALID}")
    write_skill(tmp_path / "b" / "two", frontmatter=f"name: two\ndescription: {VALID}")

    assert sorted(s.name for s in load_skills([tmp_path])) == ["one", "two"]


def test_skips_dotted_and_vendored_directories(tmp_path: Path) -> None:
    write_skill(tmp_path / ".hidden" / "a", frontmatter=f"name: a\ndescription: {VALID}")
    write_skill(tmp_path / "node_modules" / "b", frontmatter=f"name: b\ndescription: {VALID}")
    assert load_skills([tmp_path]) == []


def test_depth_guard_stops_the_walk(tmp_path: Path) -> None:
    deep = tmp_path.joinpath(*[f"d{i}" for i in range(9)])
    write_skill(deep, frontmatter=f"description: {VALID}")
    assert load_skills([tmp_path]) == []


def test_missing_source_warns_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_skill(tmp_path / "deploy")
    with caplog.at_level(logging.WARNING, logger="midge.skills"):
        skills = load_skills([tmp_path / "nope", tmp_path])

    assert [s.name for s in skills] == ["deploy"]
    assert any("skill_source_not_found" in r.getMessage() for r in caplog.records)


def test_explicit_markdown_file_source(tmp_path: Path) -> None:
    path = write_skill(tmp_path / "deploy")
    (skill,) = load_skills([path])
    assert skill.name == "deploy"


def test_non_markdown_file_source_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    junk = tmp_path / "notes.txt"
    junk.write_text("hi", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="midge.skills"):
        assert load_skills([junk]) == []
    assert any("skill_source_not_markdown" in r.getMessage() for r in caplog.records)


def test_broken_symlink_is_skipped(tmp_path: Path) -> None:
    write_skill(tmp_path / "deploy")
    os.symlink(tmp_path / "missing", tmp_path / "dangling")
    assert [s.name for s in load_skills([tmp_path])] == ["deploy"]


def test_symlinked_duplicate_dedupes_without_a_collision_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    real = tmp_path / "real"
    write_skill(real / "deploy")
    link = tmp_path / "link"
    os.symlink(real, link)

    with caplog.at_level(logging.WARNING, logger="midge.skills"):
        skills = load_skills([real, link])

    assert len(skills) == 1
    assert not any("skill_name_shadowed" in r.getMessage() for r in caplog.records)


def test_first_source_wins_a_name_collision(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_skill(tmp_path / "first" / "deploy", body="from first")
    write_skill(tmp_path / "second" / "deploy", body="from second")

    with caplog.at_level(logging.WARNING, logger="midge.skills"):
        (skill,) = load_skills([tmp_path / "first", tmp_path / "second"])

    assert skill.path == (tmp_path / "first" / "deploy" / "SKILL.md").resolve()
    warning = next(
        r.getMessage() for r in caplog.records if "skill_name_shadowed" in r.getMessage()
    )
    assert "second" in warning and "first" in warning


def test_missing_description_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_skill(tmp_path / "deploy", frontmatter="name: deploy")
    with caplog.at_level(logging.WARNING, logger="midge.skills"):
        assert load_skills([tmp_path]) == []
    assert any("skill_description_missing" in r.getMessage() for r in caplog.records)


def test_blank_description_is_skipped(tmp_path: Path) -> None:
    write_skill(tmp_path / "deploy", frontmatter='name: deploy\ndescription: "   "')
    assert load_skills([tmp_path]) == []


def test_no_frontmatter_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    d = tmp_path / "deploy"
    d.mkdir()
    (d / "SKILL.md").write_text("# Just markdown\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="midge.skills"):
        assert load_skills([tmp_path]) == []
    assert any("skill_frontmatter_missing" in r.getMessage() for r in caplog.records)


def test_malformed_yaml_is_skipped_and_siblings_still_load(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # An unquoted colon-space inside a value is a YAML syntax error.
    write_skill(tmp_path / "bad", frontmatter="name: bad\ndescription: Use when: X")
    write_skill(tmp_path / "good", frontmatter=f"name: good\ndescription: {VALID}")

    with caplog.at_level(logging.WARNING, logger="midge.skills"):
        skills = load_skills([tmp_path])

    assert [s.name for s in skills] == ["good"]
    assert any("skill_frontmatter_invalid" in r.getMessage() for r in caplog.records)


def test_long_description_warns_but_loads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_skill(tmp_path / "deploy", frontmatter="name: deploy\ndescription: " + "x" * 1100)
    with caplog.at_level(logging.WARNING, logger="midge.skills"):
        assert len(load_skills([tmp_path])) == 1
    assert any("skill_description_too_long" in r.getMessage() and "chars=1100" in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize(
    ("name", "fragment"),
    [
        ("x" * 70, "skill_name_too_long"),
        ("Deploy_Thing", "skill_name_bad_charset"),
        ("-deploy", "skill_name_edge_hyphen"),
        ("deploy-", "skill_name_edge_hyphen"),
        ("deploy--now", "skill_name_double_hyphen"),
    ],
)
def test_bad_names_warn_but_load(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, name: str, fragment: str
) -> None:
    write_skill(tmp_path / "d", frontmatter=f"name: {name}\ndescription: {VALID}")
    with caplog.at_level(logging.WARNING, logger="midge.skills"):
        (skill,) = load_skills([tmp_path])

    assert skill.name == name
    assert any(fragment in r.getMessage() for r in caplog.records)


def test_non_string_name_falls_back(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Unquoted `on` is a YAML boolean, not the string the author meant.
    write_skill(tmp_path / "toggles", frontmatter=f"name: on\ndescription: {VALID}")
    with caplog.at_level(logging.WARNING, logger="midge.skills"):
        (skill,) = load_skills([tmp_path])

    assert skill.name == "toggles"
    assert any("skill_name_not_a_string" in r.getMessage() for r in caplog.records)


def test_unknown_frontmatter_keys_are_ignored(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_skill(
        tmp_path / "deploy",
        frontmatter=(
            f"name: deploy\ndescription: {VALID}\n"
            "license: MIT\nallowed-tools: read bash\nmetadata:\n  team: infra"
        ),
    )
    with caplog.at_level(logging.WARNING, logger="midge.skills"):
        (skill,) = load_skills([tmp_path])

    assert skill.name == "deploy"
    assert caplog.records == []


def test_multiline_folded_description(tmp_path: Path) -> None:
    # The case a hand-rolled `key: value` parser gets wrong, and the reason
    # this module depends on a real YAML parser.
    write_skill(
        tmp_path / "deploy",
        frontmatter="name: deploy\ndescription: >-\n  Use when the user asks\n  about releases.",
    )
    (skill,) = load_skills([tmp_path])
    assert skill.description == "Use when the user asks about releases."


def test_literal_block_description(tmp_path: Path) -> None:
    write_skill(
        tmp_path / "deploy",
        frontmatter="name: deploy\ndescription: |\n  Line one.\n  Line two.",
    )
    (skill,) = load_skills([tmp_path])
    assert skill.description == "Line one.\nLine two."


def test_default_skill_dirs_are_absolute_and_project_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    home = Path.home()

    assert default_skill_dirs() == [
        tmp_path / ".midge" / "skills",
        tmp_path / ".agents" / "skills",
        home / ".midge" / "skills",
        home / ".agents" / "skills",
    ]


def test_skills_prompt_empty_when_no_skills() -> None:
    assert skills_prompt([]) == ""


def test_skills_prompt_lists_metadata(tmp_path: Path) -> None:
    write_skill(tmp_path / "deploy")
    skills = load_skills([tmp_path])
    prompt = skills_prompt(skills)

    assert "<available_skills>" in prompt and "</available_skills>" in prompt
    assert "<name>deploy</name>" in prompt
    assert f"<description>{VALID}</description>" in prompt
    assert f"<location>{skills[0].path}</location>" in prompt
    assert "read tool" in prompt


def test_skills_prompt_escapes_xml(tmp_path: Path) -> None:
    write_skill(tmp_path / "deploy", frontmatter='name: deploy\ndescription: "A & B < C"')
    prompt = skills_prompt(load_skills([tmp_path]))

    assert "A &amp; B &lt; C" in prompt
    assert "A & B < C" not in prompt


def test_disable_model_invocation_hides_from_catalogue_but_stays_loaded(
    tmp_path: Path,
) -> None:
    write_skill(
        tmp_path / "manual",
        frontmatter=f"name: manual\ndescription: {VALID}\ndisable-model-invocation: true",
    )
    skills = load_skills([tmp_path])

    assert [s.name for s in skills] == ["manual"]
    assert skills[0].model_invocable is False
    assert skills_prompt(skills) == ""
    assert find_skill(skills, "manual") is not None


def test_find_skill_returns_none_for_unknown() -> None:
    assert find_skill([], "nope") is None


def test_skill_message_wraps_the_body(tmp_path: Path) -> None:
    write_skill(tmp_path / "deploy", body="# Deploy\n\nRun the checklist.")
    skills = load_skills([tmp_path])
    msg = skill_message(skills, "deploy")

    assert isinstance(msg, UserMessage)
    assert isinstance(msg.content, str)
    assert msg.content.startswith(f'<skill name="deploy" location="{skills[0].path}">')
    assert f"References are relative to {skills[0].base_dir}." in msg.content
    assert "Run the checklist." in msg.content
    assert msg.content.endswith("</skill>")
    # Frontmatter is stripped, not forwarded.
    assert "description:" not in msg.content


def test_skill_message_appends_instructions(tmp_path: Path) -> None:
    write_skill(tmp_path / "deploy")
    msg = skill_message(load_skills([tmp_path]), "deploy", "target staging")
    assert isinstance(msg.content, str)
    assert msg.content.endswith("</skill>\n\ntarget staging")


def test_skill_message_unknown_name_raises(tmp_path: Path) -> None:
    write_skill(tmp_path / "deploy")
    with pytest.raises(KeyError, match="nope"):
        skill_message(load_skills([tmp_path]), "nope")


def test_skill_message_reads_the_body_at_invocation_time(tmp_path: Path) -> None:
    path = write_skill(tmp_path / "deploy", body="original")
    skills = load_skills([tmp_path])
    path.write_text(
        f"---\nname: deploy\ndescription: {VALID}\n---\n\nrewritten\n", encoding="utf-8"
    )

    msg = skill_message(skills, "deploy")
    assert isinstance(msg.content, str)
    assert "rewritten" in msg.content


async def test_forced_skill_reaches_history_without_agent_changes(tmp_path: Path) -> None:
    """`Agent.stream` already takes a `UserMessage`, so forcing needs no new seam."""
    write_skill(tmp_path / "deploy", body="Follow the checklist.")
    skills = load_skills([tmp_path])

    client = Client()

    install(client, [[say("ok"), finish()]])
    agent = Agent(client=client, model="gpt-4o")
    await agent.run(skill_message(skills, "deploy"))

    first = agent.history[0]
    assert isinstance(first, UserMessage)
    assert isinstance(first.content, str)
    assert first.content.startswith('<skill name="deploy"')
    assert "Follow the checklist." in first.content


def _run_cli(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> Agent:
    """Drive `cli.main` far enough to inspect the composed agent."""
    captured: list[Agent] = []
    # `run_tui` takes the `Controls` both front-ends share; the agent is on it.
    monkeypatch.setattr(cli, "run_tui", lambda controls, **kw: captured.append(controls.agent))
    cli.main(argv)
    return captured[0]


def test_cli_appends_the_catalogue_to_the_system_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_skill(tmp_path / "packs" / "deploy")

    agent = _run_cli(["--skill-dir", str(tmp_path / "packs")], monkeypatch)

    assert agent.system_prompt is not None
    assert agent.system_prompt.startswith(cli.BASE_SYSTEM_PROMPT)
    assert "<available_skills>" in agent.system_prompt
    assert "<name>deploy</name>" in agent.system_prompt


def test_cli_omits_the_catalogue_without_a_read_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_skill(tmp_path / "packs" / "deploy")
    # Retargeting away from the coding pack is the case this gate exists for:
    # advertising skills the model cannot open is worse than saying nothing.
    monkeypatch.setattr(cli, "BUILTIN_TOOL_DIRS", [])

    agent = _run_cli(["--skill-dir", str(tmp_path / "packs")], monkeypatch)

    assert agent.system_prompt is not None
    assert "available_skills" not in agent.system_prompt


def test_cli_recomposes_the_catalogue_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    packs = tmp_path / "packs"
    write_skill(packs / "deploy")
    session = tmp_path / "s.jsonl"

    _run_cli(["--skill-dir", str(packs), "--session", str(session)], monkeypatch)

    # A skill added after the session began must still show up on resume.
    write_skill(packs / "rollback", frontmatter=f"name: rollback\ndescription: {VALID}")
    agent = _run_cli(["--skill-dir", str(packs), "--session", str(session)], monkeypatch)

    assert agent.system_prompt is not None
    assert "<name>rollback</name>" in agent.system_prompt


def test_skill_is_frozen(tmp_path: Path) -> None:
    write_skill(tmp_path / "deploy")
    (skill,) = load_skills([tmp_path])
    with pytest.raises(AttributeError):
        skill.name = "other"  # type: ignore[misc]
    assert isinstance(skill, Skill)
