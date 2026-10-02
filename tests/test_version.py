"""__version__ comes from pyproject.toml, not a hand-written string."""

from __future__ import annotations

import tomllib
from pathlib import Path

from midge import __version__

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_version_matches_pyproject() -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    assert __version__ == data["project"]["version"]
