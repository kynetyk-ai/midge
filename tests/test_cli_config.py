"""The flag layer of `flag > env > config > default`.

`main` builds a Client and starts a UI, so it is not called here. What is worth
pinning is the one thing that silently disables the config layer if it regresses:
an argparse default that is not `None`.
"""

from __future__ import annotations

from pathlib import Path

from midge.cli import BASE_SYSTEM_PROMPT, _parse_args, resume_identity
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
    assert Config().compaction_keep_recent == DEFAULT_KEEP_RECENT


def test_a_given_flag_carries_its_value() -> None:
    args = _parse_args(["--compaction-keep-recent", "5000", "--compaction-threshold", "120000"])
    assert args.compaction_keep_recent == 5000
    assert args.compaction_threshold == 120_000


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
