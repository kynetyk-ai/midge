"""The flag layer of `flag > env > config > default`.

`main` builds a Client and starts a UI, so it is not called here. What is worth
pinning is the one thing that silently disables the config layer if it regresses:
an argparse default that is not `None`.
"""

from __future__ import annotations

from midge.cli import _parse_args
from midge.config import DEFAULT_KEEP_RECENT, Config


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
