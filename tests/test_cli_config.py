"""The flag layer of `flag > env > config > default`.

`main` builds a Client and starts a UI, so it is only called here for the
startup refusals, which happen before either. What is otherwise worth pinning is
the one thing that silently disables the config layer if it regresses: an
argparse default that is not `None`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from midge.cli import BASE_SYSTEM_PROMPT, _parse_args, main, resume_identity
from midge.config import DEFAULT_KEEP_RECENT, Config
from midge.persistence import Session


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
    assert Config().compaction_keep_recent == DEFAULT_KEEP_RECENT
    assert Config().default_profile is None


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

    assert resume_identity(Session.load(path)) == ("switched-to", "adversarial reviewer")


def test_resume_falls_back_to_the_header_and_then_the_built_in(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    with Session.new(path, model="m", system_prompt="from-header"):
        pass
    assert resume_identity(Session.load(path)) == ("m", "from-header")

    bare = tmp_path / "b.jsonl"
    with Session.new(bare, model="m"):
        pass
    assert resume_identity(Session.load(bare)) == ("m", BASE_SYSTEM_PROMPT)


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


def test_an_unknown_profile_name_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--extension-dir", str(tmp_path), "--profile", "nope"])
    assert "was not discovered" in str(excinfo.value)


def test_a_profile_that_failed_validation_says_so(tmp_path: Path) -> None:
    (tmp_path / "rev.py").write_text(_BROKEN)
    with pytest.raises(SystemExit) as excinfo:
        main(["--extension-dir", str(tmp_path), "--profile", "rev"])
    message = str(excinfo.value)
    assert "failed validation" in message
    assert "was not discovered" not in message
