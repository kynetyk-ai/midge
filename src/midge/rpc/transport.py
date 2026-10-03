"""The stdio binding, and the pipe-shaped decisions that come with it.

`serve(read_line=, write=)` takes callables rather than a stream, so the loop is
transport-agnostic — which is what lets the whole protocol be tested in-process
without pipes. `serve_stdio` is one binding of that seam, and it owns the
pipe-shaped decisions: `READ_LIMIT` exists because the default 64 KiB would
refuse a large pasted prompt, `read_bounded_line` makes a line past it a refusal
rather than a crash, and `_stdout_writer` suspends rather than blocks because a
pipe holds only ~64 KiB.

**midge never listens on anything.** There is no socket, no port and no bind
address anywhere in the package; the only network traffic is outbound, to the
model provider. That is a property worth keeping rather than an omission: stdin
and stdout are a *capability* handed to the process by whoever launched it, so
access control is inherited from the OS and the container runtime and there is
no authentication to write, no bind address to get wrong, and no way to expose
the agent by accident. LSP and MCP make the same choice.

Bridging to a socket is deliberately left to whoever deploys midge, because the
right shape is decided by the client — a stdio pipe for an editor extension that
already spawns processes, a WebSocket for a UI that needs server-push, a queue
for anything request/response. Those are different concurrency models, not
variations on a transport. What such a bridge inherits from here, and has to
decide for itself:

- **Anything that can send a line can run `bash`** with this process's
  privileges. There is no notion of a caller and no authorization layer; the
  protocol assumes the peer is already trusted. **The boundary is the container
  or the OS user, not midge.** A `tool_call` hook gates *tool calls*, not their
  effects: it can refuse to run `bash`, or restrict `write` to a directory, but
  a policy that reads a `bash` command string is advisory — `rm -rf` blocked is
  `find -delete` allowed, and a guard on `write`'s path never sees a shell
  redirect (#102). Hooks apply to sub-agents too. For a real restriction, leave
  `bash` out of the tool set: `examples/allowlist_extension/`.
- **One client, one agent, one session, one process.** A second client would
  share the same conversation and the same history. Multi-tenancy means multiple
  processes.
- **EOF terminates the loop** (`if not line: break`). Over a pipe that is right —
  the parent is gone, so should we be. Over a socket it is a decision: a client
  disconnecting mid-task probably should not kill the agent.
- **`READ_LIMIT` and the writer's backpressure behaviour are tuned for pipes**
  and should be revisited for a transport with different framing and buffering.

Nothing here is a limit of the protocol; it is what the protocol currently
assumes, recorded so a bridge author does not have to rediscover it.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import signal
import stat
import sys
import threading
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from midge.rpc.server import RpcServer

_logger = logging.getLogger(__name__)

READ_LIMIT = 16 * 1024 * 1024
# Roughly ten ordinary answers' worth of frames — deep enough that a client
# pausing to render or collect garbage never stalls anything, shallow enough
# that a client which has died applies backpressure instead of exhausting memory.
OUTBOX_FRAMES = 4096
FLUSH_TIMEOUT = 5.0

WriteFn = Callable[[bytes], Awaitable[None]]


class LineTooLong(Exception):
    """A command line longer than the reader's limit, already discarded.

    Its own type so `serve` can answer it and keep going without catching a
    general `ValueError` — which is what `StreamReader.readline` raises, and
    what used to escape `serve` and end the process (#100).
    """

    def __init__(self, size: int, limit: int) -> None:
        super().__init__(f"line of {size} bytes exceeds the {limit // (1024 * 1024)} MiB limit")
        self.size = size
        self.limit = limit


async def read_bounded_line(reader: asyncio.StreamReader, limit: int = READ_LIMIT) -> bytes:
    """One line from `reader`, or `LineTooLong` with the whole line consumed.

    `limit` is only what the refusal reports; the reader enforces its own.
    `readline()` is not used because of what it does past the limit: it clears
    what it has buffered and raises, while the rest of the line is still
    arriving — so the *next* read returns that tail as if it were a command.
    This drains through the newline first, so the stream is back on a line
    boundary when the refusal goes out.
    """
    try:
        return await reader.readuntil(b"\n")
    except asyncio.IncompleteReadError as e:
        return e.partial  # EOF mid-line: the same as `readline`
    except asyncio.LimitOverrunError as e:
        dropped = 0
        overrun: asyncio.LimitOverrunError | None = e
        while overrun is not None:
            dropped += len(await reader.read(overrun.consumed))
            try:
                dropped += len(await reader.readuntil(b"\n"))
                overrun = None
            except asyncio.LimitOverrunError as again:
                overrun = again
            except asyncio.IncompleteReadError as eof:
                dropped += len(eof.partial)
                overrun = None
        raise LineTooLong(dropped, limit) from None
ReadLineFn = Callable[[], Awaitable[bytes]]

_claimed_stdout: BinaryIO | None = None
# The wrapper we displace is kept alive deliberately: dropping the last
# reference to a TextIOWrapper closes the buffer underneath it, which here is
# the fd carrying the protocol. In a real process `sys.__stdout__` happens to
# hold one too, but relying on that is a footgun.
_displaced_stdout: Any = None


def claim_stdout() -> BinaryIO:
    """Take fd 1 for the protocol and point `sys.stdout` at stderr.

    Stdout is the wire here, so a single stray `print()` — from a tool, a hook,
    a user extension, or a dependency — corrupts it. The corruption is quiet
    rather than loud: the protocol writes through the buffered binary layer
    while `print` goes through the text wrapper above it, so under a pipe the
    stray text is block-buffered and surfaces at some arbitrary later point.
    Individual frames stay intact; their ordering does not.

    Returns the real stdout for the protocol writer to hold. Idempotent.
    """
    global _claimed_stdout, _displaced_stdout
    if _claimed_stdout is not None:
        return _claimed_stdout

    _displaced_stdout = sys.stdout
    real = sys.stdout.buffer
    sys.stdout = io.TextIOWrapper(
        sys.stderr.buffer, encoding=sys.stderr.encoding, errors="replace", line_buffering=True
    )
    _claimed_stdout = real
    return real


async def _connect_stdin_reader(reader: asyncio.StreamReader) -> None:
    """Connect stdin to `reader`, using a thread when the selector cannot register it.

    The selector refuses character devices such as `/dev/null` and regular files,
    so `connect_read_pipe` raises inside an asyncio callback. In those cases a
    daemon thread feeds data to the loop instead.
    """
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    st = os.fstat(fd)
    if stat.S_ISFIFO(st.st_mode) or stat.S_ISSOCK(st.st_mode) or sys.stdin.isatty():
        await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    else:
        def _feed() -> None:
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    loop.call_soon_threadsafe(reader.feed_eof)
                    break
                loop.call_soon_threadsafe(reader.feed_data, chunk)

        threading.Thread(target=_feed, daemon=True).start()


async def serve_stdio(server: RpcServer) -> None:
    """Run `server` over this process's stdin/stdout.

    Claims stdout first, installs SIGTERM/SIGHUP handlers so a supervisor can
    stop the process cleanly, and shuts down on stdin EOF.
    """
    stdout = claim_stdout()
    loop = asyncio.get_running_loop()

    # The default 64 KiB would refuse a large pasted prompt. Raising it moved
    # the cliff rather than removing it; `read_bounded_line` is what removes it.
    reader = asyncio.StreamReader(limit=READ_LIMIT)
    await _connect_stdin_reader(reader)

    async def read_line() -> bytes:
        return await read_bounded_line(reader)

    write, close_writer = await _stdout_writer(loop, stdout)

    serving = asyncio.ensure_future(server.serve(read_line=read_line, write=write))

    def _stop(signame: str) -> None:
        _logger.info("rpc_signal signal=%s", signame)
        serving.cancel()

    installed: list[signal.Signals] = []
    for sig in (signal.SIGTERM, signal.SIGHUP):
        with contextlib.suppress(NotImplementedError, AttributeError):
            loop.add_signal_handler(sig, _stop, sig.name)
            installed.append(sig)
    try:
        # Cancelling is a clean stop here, not a failure: `serve`'s own finally
        # cancels the in-flight run on the way out.
        with contextlib.suppress(asyncio.CancelledError):
            await serving
    finally:
        for sig in installed:
            with contextlib.suppress(NotImplementedError):
                loop.remove_signal_handler(sig)
        close_writer()



async def _stdout_writer(
    loop: asyncio.AbstractEventLoop, stdout: BinaryIO
) -> tuple[WriteFn, Callable[[], None]]:
    """A writer that suspends rather than blocks when the client stops reading.

    A pipe holds ~64 KiB, and one ordinary assistant answer is ~20 KiB of frames
    because every token is its own record — so three answers fill it. Writing
    with a plain `file.write` then blocks the *event loop*, which stops the
    agent, every tool, and the stdin reader together. `abort` cannot get through
    because it arrives on the blocked reader, and the SIGTERM handler cannot run
    because it is queued on the blocked loop. Only SIGKILL is left.

    `drain()` suspends the calling coroutine instead, so the loop keeps
    scheduling: commands are still dispatched and an abort still lands. The
    agent throttles to the speed of the client, which is the correct answer —
    pausing beats both blocking and buffering without bound.
    """
    try:
        transport, protocol = await loop.connect_write_pipe(
            lambda: asyncio.streams.FlowControlMixin(loop), stdout
        )
    except ValueError:
        # Not a pipe, socket or tty — `midge --rpc > out.jsonl`. A regular file
        # has no reader to stall behind, so blocking writes are fine here and
        # asyncio refuses to wrap it anyway.
        _logger.debug("rpc_writer mode=blocking reason=not_a_pipe")

        async def write_blocking(data: bytes) -> None:
            stdout.write(data)
            stdout.flush()

        return write_blocking, lambda: None

    writer = asyncio.StreamWriter(transport, protocol, None, loop)

    async def write(data: bytes) -> None:
        writer.write(data)
        await writer.drain()

    return write, transport.close
