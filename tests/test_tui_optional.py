"""Textual is optional: `midge --rpc` without it, and TUI startup with a message."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import tests.fake_openai as fake_openai_mod

pytestmark = pytest.mark.subprocess

_BLOCKER = (
    "import sys\n"
    "class _Block:\n"
    "    def find_spec(self, name, *a, **kw):\n"
    "        if name == 'textual' or name.startswith('textual.'):\n"
    "            raise ImportError(name)\n"
    "        return None\n"
    "sys.meta_path.insert(0, _Block())\n"
)


def test_rpc_without_textual(tmp_path: Path) -> None:
    """`midge --rpc --no-session` starts without Textual."""
    with fake_openai_mod.FakeOpenAI([]) as fake:
        code = _BLOCKER + (
            "from midge.cli import main\n"
            'main(["--rpc", "--no-session"])\n'
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "MIDGE_"))}
        env |= {
            "OPENAI_BASE_URL": fake.base_url,
            "MIDGE_MODEL": "m",
            "HOME": str(tmp_path),
        }
        proc = subprocess.run(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            capture_output=True,
            timeout=30,
            env=env,
            cwd=tmp_path,
        )
        assert proc.returncode == 0, proc.stderr
        first_line = proc.stdout.splitlines()[0]
        frame: dict[str, Any] = json.loads(first_line)
        assert frame["type"] == "ready"


def test_tui_without_textual(tmp_path: Path) -> None:
    """Running the TUI without Textual exits 2 with a helpful message."""
    code = _BLOCKER + (
        "from midge.cli import main\n"
        "main([])\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "MIDGE_"))}
    env |= {
        "MIDGE_MODEL": "m",
        "HOME": str(tmp_path),
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        cwd=tmp_path,
    )
    assert proc.returncode == 2
    assert "midge[tui]" in proc.stderr
    assert "Traceback" not in proc.stderr
