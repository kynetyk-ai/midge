"""JSON-over-stdio RPC server for embedding the agent in external tools.

**The protocol is documented in `docs/rpc.md`** — every command, response and
event, the ordering guarantees, and the compatibility policy behind
`PROTOCOL_VERSION`. It is pinned by `tests/test_rpc_contract.py`, so a change to
a frame's shape fails there until the reference and the golden file follow.

Stdout is the protocol; stderr is for diagnostics. Call `claim_stdout()` before
anything else can write, so a stray `print()` anywhere in the process lands on
stderr instead of corrupting the stream.

The package is four modules: `wire` maps internal events to frames, `server`
owns the dispatch loop and the handlers, `transport`
binds the loop to stdio and records what a bridge
to anything else would have to decide, and `client`
is a standard-library client for hosts.
"""

from midge.rpc.server import (
    BUILTIN_COMMANDS,
    PROTOCOL_VERSION,
    RELOAD_TARGETS,
    SKILL_COMMAND_PREFIX,
    TRANSCRIPT_OPTIONS,
    BuiltinCommand,
    RpcServer,
)
from midge.rpc.transport import (
    FLUSH_TIMEOUT,
    OUTBOX_FRAMES,
    READ_LIMIT,
    LineTooLong,
    ReadLineFn,
    WriteFn,
    _stdout_writer,
    claim_stdout,
    read_bounded_line,
    serve_stdio,
)
from midge.rpc.wire import event_to_wire

__all__ = [
    "BUILTIN_COMMANDS",
    "FLUSH_TIMEOUT",
    "OUTBOX_FRAMES",
    "PROTOCOL_VERSION",
    "READ_LIMIT",
    "RELOAD_TARGETS",
    "SKILL_COMMAND_PREFIX",
    "TRANSCRIPT_OPTIONS",
    "BuiltinCommand",
    "LineTooLong",
    "ReadLineFn",
    "RpcServer",
    "WriteFn",
    # Underscored but re-exported: `tests/test_rpc_stdout.py` drives the writer
    # directly, which is the only way to test backpressure without a real pipe.
    "_stdout_writer",
    "claim_stdout",
    "event_to_wire",
    "read_bounded_line",
    "serve_stdio",
]
