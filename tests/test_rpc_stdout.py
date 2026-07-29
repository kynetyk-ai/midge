from __future__ import annotations

import io
import sys
from collections.abc import Iterator

import pytest

import midge.rpc as rpc


@pytest.fixture(autouse=True)
def _restore_streams() -> Iterator[None]:
    saved_out = sys.stdout
    saved = (rpc._claimed_stdout, rpc._displaced_stdout)
    rpc._claimed_stdout, rpc._displaced_stdout = None, None
    yield
    sys.stdout = saved_out
    rpc._claimed_stdout, rpc._displaced_stdout = saved


class _Streams:
    """Stand-ins for the real fds, so a test can see where bytes landed."""

    def __init__(self) -> None:
        self.out = io.BytesIO()
        self.err = io.BytesIO()
        sys.stdout = io.TextIOWrapper(self.out, encoding="utf-8")
        self._saved_err = sys.stderr
        sys.stderr = io.TextIOWrapper(self.err, encoding="utf-8")

    def flush(self) -> None:
        sys.stdout.flush()
        sys.stderr.flush()

    def restore(self) -> None:
        sys.stderr = self._saved_err


def test_print_after_claiming_goes_to_stderr() -> None:
    s = _Streams()
    try:
        protocol = rpc.claim_stdout()
        print("stray output from a tool")
        s.flush()

        assert b"stray" in s.err.getvalue()
        assert b"stray" not in s.out.getvalue()
        assert protocol is not None
    finally:
        s.restore()


def test_protocol_handle_still_reaches_real_stdout() -> None:
    s = _Streams()
    try:
        protocol = rpc.claim_stdout()
        protocol.write(b'{"type":"response"}\n')
        protocol.flush()
        s.flush()

        assert s.out.getvalue() == b'{"type":"response"}\n'
        assert b"response" not in s.err.getvalue()
    finally:
        s.restore()


def test_sys_stdout_write_also_diverted() -> None:
    s = _Streams()
    try:
        rpc.claim_stdout()
        sys.stdout.write("not framed json\n")
        s.flush()

        assert b"not framed" in s.err.getvalue()
        assert s.out.getvalue() == b""
    finally:
        s.restore()


def test_claiming_twice_returns_the_same_handle() -> None:
    s = _Streams()
    try:
        first = rpc.claim_stdout()
        diverted = sys.stdout
        second = rpc.claim_stdout()

        assert first is second
        assert sys.stdout is diverted, "a second claim must not re-wrap stderr"
    finally:
        s.restore()


def test_interleaving_keeps_the_protocol_clean() -> None:
    """The realistic failure: a tool printing between two frames."""
    s = _Streams()
    try:
        protocol = rpc.claim_stdout()
        protocol.write(b'{"seq":1}\n')
        print("chatty tool")
        protocol.write(b'{"seq":2}\n')
        protocol.flush()
        s.flush()

        assert s.out.getvalue() == b'{"seq":1}\n{"seq":2}\n'
    finally:
        s.restore()
