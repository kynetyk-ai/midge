from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from midge import persistence


@pytest.fixture(autouse=True)
def _isolated_session_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Keep the suite out of the developer's real `.midge/sessions`.

    `default_session_dir()` is `cwd/.midge/sessions`, so anything that resolves a
    session path or lists sessions without being told where reads whatever
    happens to be on the machine running the tests. That makes a listing test
    pass or fail depending on what you did yesterday — and a test that opens one
    of those files is reading a real conversation.

    Patched for every test rather than the few that noticed, because the ones
    that notice are exactly the ones already passing an explicit directory.
    """
    root = tmp_path_factory.mktemp("sessions")
    monkeypatch.setattr(persistence, "default_session_dir", lambda: root)
    yield root
