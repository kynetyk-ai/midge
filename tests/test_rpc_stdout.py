from __future__ import annotations

import asyncio
import io
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import midge.rpc as rpc
from tests.fakes import install_gated


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


# ---- backpressure ----


async def test_a_stalled_writer_does_not_stop_the_dispatch_loop() -> None:
    """The property that matters: a client that stops reading must not cost
    the server its ability to be told to stop.

    With a blocking writer the event loop itself stalls, so `abort` cannot be
    read and the SIGTERM handler cannot run — only SIGKILL is left.
    """
    from midge.agent import Agent
    from midge.client import Client
    from midge.rpc import RpcServer

    released = asyncio.Event()
    written: list[bytes] = []

    async def stalled_write(data: bytes) -> None:
        written.append(data)
        await released.wait()

    inbox: asyncio.Queue[bytes] = asyncio.Queue()

    async def read_line() -> bytes:
        return await inbox.get()

    client = Client()
    gate = asyncio.Event()
    # A turn that emits nothing and hangs, so the only frames on the wire are the
    # ones the server itself writes — which is what the stalled writer blocks on.
    install_gated(client, [], gate)
    agent = Agent(client=client, model="m")
    server = RpcServer(agent)
    task = asyncio.create_task(server.serve(read_line=read_line, write=stalled_write))

    inbox.put_nowait(b'{"id":"p","type":"prompt","message":"hi"}\n')
    await asyncio.sleep(0.05)
    run = server._current_run
    assert run is not None and not run.done()

    # The writer is stuck on the very first frame.
    assert len(written) == 1

    # Abort must still be dispatched and must still cancel the run, even though
    # its own response cannot be written yet.
    inbox.put_nowait(b'{"id":"a","type":"abort"}\n')
    for _ in range(50):
        await asyncio.sleep(0.01)
        if run.cancelled() or run.done():
            break
    assert run.cancelled() or run.done(), "abort never reached the run"

    released.set()
    gate.set()
    inbox.put_nowait(b"")
    # Bounded so a regression fails the test rather than hanging the suite.
    await asyncio.wait_for(task, timeout=5)


def test_writer_falls_back_to_blocking_for_a_regular_file(tmp_path: Path) -> None:
    """`midge --rpc > out.jsonl`: asyncio refuses to wrap a regular file, and a
    file has no reader to stall behind, so blocking writes are correct there."""
    from midge.rpc import _stdout_writer

    async def go() -> None:
        path = tmp_path / "out.jsonl"
        with path.open("wb") as fh:
            write, close = await _stdout_writer(asyncio.get_running_loop(), fh)
            await write(b'{"type":"response"}\n')
            close()
        assert path.read_bytes() == b'{"type":"response"}\n'

    asyncio.run(go())


def test_writer_uses_a_draining_transport_for_a_pipe() -> None:
    """The pipe case is the one that used to wedge the loop."""
    from midge.rpc import _stdout_writer

    async def go() -> None:
        r_fd, w_fd = os.pipe()
        reader = os.fdopen(r_fd, "rb")
        with os.fdopen(w_fd, "wb", buffering=0) as w:
            write, close = await _stdout_writer(asyncio.get_running_loop(), w)

            ticks = 0

            async def heartbeat() -> None:
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.005)
                    ticks += 1

            beat = asyncio.ensure_future(heartbeat())
            # More than a pipe buffer, so a blocking writer would wedge here.
            big = asyncio.ensure_future(write(b"x" * 300_000))
            await asyncio.sleep(0.1)

            assert ticks > 0, "the event loop stopped scheduling while writing"
            assert not big.done(), "expected the write to be suspended, not done"

            # Draining lets it finish.
            await asyncio.get_running_loop().run_in_executor(None, reader.read, 300_000)
            await asyncio.wait_for(big, timeout=5)
            beat.cancel()
            close()
        reader.close()

    asyncio.run(go())
