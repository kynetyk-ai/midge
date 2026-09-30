"""Every tunable toybox has, in one place.

The rule here is the same one most projects converge on: a value that would
differ between two machines or two deployments belongs on `Settings` with a
default, not as a literal halfway down a module. Adding one is three edits —
the field, its line in `load`, and its entry in `settings.example.toml`.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WIDTH = 72


@dataclass(frozen=True, slots=True)
class Settings:
    # How wide `text.wrap` breaks lines.
    width: int = DEFAULT_WIDTH
    # Where `tally` writes its cache. None means no cache is kept.
    cache_dir: Path | None = None
    # Whether to count words case-sensitively.
    case_sensitive: bool = False


def load(path: Path | None = None) -> Settings:
    """Read `settings.toml`, falling back to defaults for anything absent.

    A malformed file degrades to defaults rather than raising: a typo should not
    stop the program from starting.
    """
    p = path or Path("settings.toml")
    data: dict[str, object] = {}
    if p.exists():
        try:
            data = tomllib.loads(p.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            data = {}

    raw_width = os.getenv("TOYBOX_WIDTH") or data.get("width")
    try:
        width = int(raw_width) if raw_width is not None else DEFAULT_WIDTH
    except (TypeError, ValueError):
        width = DEFAULT_WIDTH

    cache = data.get("cache_dir")
    return Settings(
        width=width,
        cache_dir=Path(str(cache)) if cache else None,
        case_sensitive=bool(data.get("case_sensitive", False)),
    )
