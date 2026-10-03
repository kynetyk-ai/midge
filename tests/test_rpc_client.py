"""midge.rpc.client: the stdlib-only RPC client."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from midge.rpc.client import (
    SUPPORTED_PROTOCOLS,
    CommandError,
    MidgeClient,
    ProtocolError,
)
from tests.fake_openai import FakeOpenAI

pytestmark = pytest.mark.subprocess

TIMEOUT = 30.0


def _env(tmp_path: Path, fake: FakeOpenAI) -> dict[str, str]:
    base = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "MIDGE_"))}
    return base | {
         "OPENAI_BASE_URL": fake.base_url,
         "MIDGE_MODEL": "m",
         "HOME": str(tmp_path),
     }


def _cmd() -> list[str]:
    return [sys.executable, "-m", "midge", "--rpc", "--no-session"]


def test_the_client_reads_ready(tmp_path: Path) -> None:
    with FakeOpenAI([]) as fake, MidgeClient(
         _cmd(), cwd=tmp_path, env=_env(tmp_path, fake), timeout=TIMEOUT
     ) as client:
        assert client.protocol == 1
        assert isinstance(client.version, str) and client.version
        assert client.close() == 0


def test_get_state_matches_ready(tmp_path: Path) -> None:
    with FakeOpenAI([]) as fake, MidgeClient(
         _cmd(), cwd=tmp_path, env=_env(tmp_path, fake), timeout=TIMEOUT
     ) as client:
        state = client.get_state()
        assert state["protocol"] == client.protocol
        assert state["midge"] == client.version


def test_a_refused_command_raises(tmp_path: Path) -> None:
    with FakeOpenAI([]) as fake, MidgeClient(
         _cmd(), cwd=tmp_path, env=_env(tmp_path, fake), timeout=TIMEOUT
     ) as client:
        with pytest.raises(CommandError) as exc:
            client.abort()
        assert exc.value.error


def test_an_unknown_profile_raises(tmp_path: Path) -> None:
    with FakeOpenAI([]) as fake, MidgeClient(
         _cmd(), cwd=tmp_path, env=_env(tmp_path, fake), timeout=TIMEOUT
     ) as client, pytest.raises(CommandError):
        client.use_profile("no-such-profile")


def test_an_unsupported_protocol_is_refused(tmp_path: Path) -> None:
    stub = tmp_path / "stub.py"
    stub.write_text(
         "import json, sys\n"
         "print(json.dumps({'type': 'ready', 'protocol': 99, 'midge': '9.9'}), flush=True)\n"
         "sys.stdin.read()\n"
     )
    with pytest.raises(ProtocolError) as exc:
        MidgeClient([sys.executable, str(stub)], timeout=TIMEOUT)
    assert "99" in str(exc.value)


def test_protocols_stay_in_step() -> None:
    from midge.rpc.server import PROTOCOL_VERSION
    assert PROTOCOL_VERSION in SUPPORTED_PROTOCOLS
