from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "loc.py"


def test_the_core_harness_is_within_its_size_budget() -> None:
    """The README claims the harness is small enough to read in a sitting.

    Run as a subprocess rather than imported so this exercises exactly what CI
    runs — the same entrypoint, the same exit code — instead of a second path
    that could drift from it.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
