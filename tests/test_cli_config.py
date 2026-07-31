"""The flag layer of `flag > env > config > default`.

`main` builds a Client and starts a UI, so it is only called here for the
startup refusals, which happen before either. What is otherwise worth pinning is
the one thing that silently disables the config layer if it regresses: an
argparse default that is not `None`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from midge import cli
from midge.agent import Agent
from midge.cli import BASE_SYSTEM_PROMPT, _parse_args, main, resume_identity
from midge.config import DEFAULT_KEEP_RECENT, Config, ProviderConfig
from midge.persistence import Session
from midge.providers import ModelRegistry


def test_unset_flags_are_none_so_config_can_win() -> None:
    """The trap this guards.

    Restoring `default=20_000` on the flag would make `[compaction] keep_recent`
    unreachable — the flag would always be set, and always beat the file. The
    default has to live in `Config`, once.
    """
    args = _parse_args([])
    assert args.compaction_keep_recent is None
    assert args.compaction_threshold is None
    assert args.profile is None
    assert args.session is None
    assert Config().compaction_keep_recent == DEFAULT_KEEP_RECENT
    assert Config().default_profile is None
    # `--no-session` is the exception the trap allows: `store_true` defaults to
    # False, but False here means "no opinion" rather than a value that would
    # shadow `[session] enabled`.
    assert args.no_session is False
    assert Config().session.enabled is True
    assert Config().session.dir is None


def test_a_given_flag_carries_its_value() -> None:
    args = _parse_args(
        ["--compaction-keep-recent", "5000", "--compaction-threshold", "120000", "--profile", "rev"]
    )
    assert args.compaction_keep_recent == 5000
    assert args.compaction_threshold == 120_000
    assert args.profile == "rev"


def test_resume_reads_the_folded_identity_not_the_header(tmp_path: Path) -> None:
    """#57: `set_model` and `set_system_prompt` used to revert on resume.

    The header is written once and never rewritten, so reading it back as
    authoritative discarded every change made during the session.
    """
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="started-with", system_prompt="original") as s:
        s.set_model("switched-to")
        s.set_system_prompt("adversarial reviewer")

    assert resume_identity(Session.load(path), configured="default-model") == (
        "switched-to",
        "adversarial reviewer",
    )


def test_resume_falls_back_to_the_header_and_then_the_built_in(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m", system_prompt="from-header"):
        pass
    assert resume_identity(Session.load(path), configured="c") == ("m", "from-header")

    bare = tmp_path / "b.jsonl"
    with Session.new(bare, model="m"):
        pass
    assert resume_identity(Session.load(bare), configured="c") == ("m", BASE_SYSTEM_PROMPT)


# --- the model is a stored prior choice, not an override ---
#
# The prompt is part of what the conversation *is*, so it always comes back.
# The model is infrastructure with its own config key, so it takes part in
# precedence: it beats a default and loses to a model asked for this run.


def test_a_model_asked_for_this_run_beats_the_recorded_one(tmp_path: Path) -> None:
    """Previously the transcript won unconditionally, so an operator who set
    `MIDGE_MODEL` and resumed had it discarded with nothing said."""
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="recorded", system_prompt="reviewer"):
        pass

    model, durable = resume_identity(
        Session.load(path), configured="asked-for", configured_explicitly=True
    )
    # The prompt still comes back: it is not a setting the operator overrode.
    assert (model, durable) == ("asked-for", "reviewer")


def test_the_recorded_model_wins_over_a_mere_default(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="recorded"):
        pass

    model, _durable = resume_identity(
        Session.load(path), configured="the-default", configured_explicitly=False
    )
    assert model == "recorded"


def test_an_override_is_announced(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="recorded"):
        pass

    with caplog.at_level(logging.WARNING, logger="midge.cli"):
        resume_identity(Session.load(path), configured="asked-for", configured_explicitly=True)
    assert "resume_model_overridden" in caplog.text


def test_a_retired_model_warns_and_degrades_rather_than_refusing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The trap this removes: a session recorded against a model the vendor has
    since retired used to make midge refuse to start, on an id nobody chose
    this run and with no flag to override it."""
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="retired-by-the-vendor"):
        pass
    registry = ModelRegistry(
        models={"current": "p"}, providers={"p": ProviderConfig(kind="openai")}
    )

    with caplog.at_level(logging.WARNING, logger="midge.cli"):
        model, _durable = resume_identity(
            Session.load(path), configured="current", registry=registry
        )
    assert model == "current"
    assert "resume_model_unregistered" in caplog.text


# --- a profile is applied at startup, not merely checked (#67) ---


_REVIEWER_EXT = (
    "from midge.profiles import Profile\n"
    "P = Profile(name='rev', description='d', prompt='You are adversarial.',\n"
    "            tools=('read',), hooks={})\n"
)


def _start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> Agent:
    """Drive `main` far enough to inspect the composed agent."""
    captured: list[Agent] = []
    monkeypatch.chdir(tmp_path)
    # `run_tui` takes the `Controls` both front-ends share; the agent is on it.
    monkeypatch.setattr(cli, "run_tui", lambda controls, **kw: captured.append(controls.agent))
    main(argv)
    return captured[0]


def test_a_selected_profile_is_applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """#61 shipped discovery and validation; nothing applied one. This is what
    made `--profile` more than a spelling check."""
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "rev.py").write_text(_REVIEWER_EXT)

    agent = _start(tmp_path, monkeypatch, ["--extension-dir", str(ext), "--profile", "rev"])

    assert "You are adversarial." in (agent.system_prompt or "")
    assert sorted(t.name for t in agent.tools) == ["read"]


def test_a_resumed_session_comes_back_under_its_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A profile is a deliberate named identity, unlike a loose model id — so
    unlike the model it is restored rather than degraded."""
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "rev.py").write_text(_REVIEWER_EXT)
    resumed = tmp_path / "prior.jsonl"
    with Session.new(resumed, model="m") as s:
        s.set_profile(name="rev", model="m", system_prompt="You are adversarial.")

    agent = _start(
        tmp_path, monkeypatch, ["--extension-dir", str(ext), "--session", str(resumed)]
    )

    assert sorted(t.name for t in agent.tools) == ["read"]


def test_a_recorded_profile_that_vanished_warns_and_carries_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Report the mismatch and let the operator reconsider, rather than
    refusing to start on a profile file someone deleted."""
    resumed = tmp_path / "prior.jsonl"
    with Session.new(resumed, model="m") as s:
        s.set_profile(name="gone", model="m", system_prompt="You are adversarial.")

    with caplog.at_level(logging.WARNING, logger="midge.cli"):
        agent = _start(tmp_path, monkeypatch, ["--session", str(resumed)])

    assert "resume_profile_unavailable" in caplog.text
    # The recorded prompt is the fallback, so the conversation still reads right.
    assert "You are adversarial." in (agent.system_prompt or "")


def test_the_flag_beats_a_recorded_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "rev.py").write_text(_REVIEWER_EXT)
    (ext / "wide.py").write_text(
        "from midge.profiles import Profile\n"
        "P = Profile(name='wide', description='d', prompt='Wide open.',\n"
        "            tools=('read', 'write'), hooks={})\n"
    )
    resumed = tmp_path / "prior.jsonl"
    with Session.new(resumed, model="m") as s:
        s.set_profile(name="rev", model="m", system_prompt="You are adversarial.")

    agent = _start(
        tmp_path,
        monkeypatch,
        ["--extension-dir", str(ext), "--session", str(resumed), "--profile", "wide"],
    )

    assert sorted(t.name for t in agent.tools) == ["read", "write"]


# --- the startup profile refusal ---
#
# Two different mistakes with two different fixes: a name that matches nothing,
# and a profile whose file is broken. They must not share a message — telling
# someone their profile "was not discovered" when it was found and then
# rejected sends them to the wrong file.

_BROKEN = (
    "from midge.profiles import Profile\n"
    "P = Profile(name='rev', description='d', prompt='go', tools=('nonexistent',))\n"
)


def test_an_unknown_profile_name_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `main` now opens a transcript before it gets this far, and the default
    # location is relative to the working directory.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(["--extension-dir", str(tmp_path), "--profile", "nope"])
    assert "was not discovered" in str(excinfo.value)


def test_a_profile_that_failed_validation_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rev.py").write_text(_BROKEN)
    with pytest.raises(SystemExit) as excinfo:
        main(["--extension-dir", str(tmp_path), "--profile", "rev"])
    message = str(excinfo.value)
    assert "failed validation" in message
    assert "was not discovered" not in message
