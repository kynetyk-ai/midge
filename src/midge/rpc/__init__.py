"""JSON-over-stdio RPC server for embedding the agent in external tools.

Wire format: newline-delimited JSON, LF-only. Each emitted record is
`json.dumps(obj, ensure_ascii=False) + "\n"`.

Two outbound shapes:
    - Responses correlated to inbound commands by an optional `id`:
      {"id": "...", "type": "response", "command": "...", "success": bool, ...}
    - Async events streamed during a prompt run:
      {"type": "assistant_text_delta" | "tool_call_start" | ... }

An event from a *nested* agent carries an `agent` envelope naming which run
produced it — `{"agent": {"agent": "explore", "agent_id": "call_1",
"parent_id": null, "depth": 1}}`. It is absent on top-level events, so a client
that ignores the key sees exactly the stream it saw before, and one that reads
it can build the tree. `agent_id` is the id of the tool call that spawned the
run, which is deliberately the same id the child's transcript records as
`parent_tool_call_id` — one scheme, not two.

Only the events that say what a delegation is *doing* are forwarded: tool
executions, errors, and its end. Text and tool-argument deltas are not, because
a child emits hundreds per turn and its prose is in its own transcript.

Stdout is the protocol; stderr is for diagnostics. Call `claim_stdout()` before
anything else can write, so a stray `print()` anywhere in the process lands on
stderr instead of corrupting the stream.

Inbound: prompt, steer, follow_up, abort, and a set of state and control
commands `get_commands` enumerates for clients that would rather discover the
surface than hardcode it — including `reload`, which re-scans skills and
extensions from disk so a long-lived process picks up edits.
Commands are dispatched serially; a `prompt` returns its response immediately
after preflight and runs the agent in a background task while the dispatch
loop continues reading stdin (so `abort` can interrupt).

The package is three modules: `wire` maps internal events to frames, `server`
owns the dispatch loop and the handlers, `transport` binds the loop to stdio.
Everything a caller needs is re-exported here.
"""

from midge.rpc.server import (
    BUILTIN_COMMANDS,
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
    ReadLineFn,
    WriteFn,
    _stdout_writer,
    claim_stdout,
    serve_stdio,
)
from midge.rpc.wire import event_to_wire

__all__ = [
    "BUILTIN_COMMANDS",
    "FLUSH_TIMEOUT",
    "OUTBOX_FRAMES",
    "READ_LIMIT",
    "RELOAD_TARGETS",
    "SKILL_COMMAND_PREFIX",
    "TRANSCRIPT_OPTIONS",
    "BuiltinCommand",
    "ReadLineFn",
    "RpcServer",
    "WriteFn",
    # Underscored but re-exported: `tests/test_rpc_stdout.py` drives the writer
    # directly, which is the only way to test backpressure without a real pipe.
    "_stdout_writer",
    "claim_stdout",
    "event_to_wire",
    "serve_stdio",
]
