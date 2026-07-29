# RPC mode (JSON-over-stdio) — patterns to borrow from `pi-mono`

Source:
- `pi-mono/packages/coding-agent/src/cli/args.ts:74–78` — `--mode rpc` flag
- `pi-mono/packages/coding-agent/src/modes/rpc/rpc-mode.ts` — entrypoint and dispatch
- `pi-mono/packages/coding-agent/src/modes/rpc/jsonl.ts` — wire framing
- `pi-mono/packages/coding-agent/src/modes/rpc/rpc-types.ts` — full type catalog
- `pi-mono/packages/coding-agent/src/modes/rpc/rpc-client.ts` — TS reference client

## What we're porting and what we're not

`pi-mono`'s RPC surface is huge — 20+ inbound command types covering steering, model swaps, fork/clone, session naming, extension UI prompts, etc. **For Phase 2 we ship a minimal subset**: enough for an external Python script to drive a single agent over stdio. Everything else can be added incrementally if a real client needs it.

**In scope (Phase 2):**
- `prompt` (inbound) — submit a user message; response begins streaming
- `abort` (inbound) — cancel current turn
- `get_messages` (inbound) — return current history
- All assistant streaming events as outbound notifications
- Tool execution events
- Errors

**Out of scope (Phase 2):**
- Steering / follow-up modes
- Model swap, thinking-level cycling
- Fork / clone / session-switch
- Extension-UI request/response (modal prompts driven from Python)
- Auto-compaction / auto-retry
- `bash` / `abort_bash` direct execution

> **Update (#30).** The Phase 2 subset above has been extended. Added since:
> `get_state`, `get_last_assistant_text`, `get_system_prompt`,
> `set_system_prompt`, `set_model`, `export_html`, `compact`, `clear_context`,
> `new_session`, plus `steer` / `follow_up` and the `agent_settled` terminal
> event. Model swap arrived as a bare `set_model`; thinking levels, fork/clone,
> the extension-UI channel, auto-retry and `bash` are all still out, for the
> reasons in #30. Session naming needs a storage decision and is filed
> separately.

## Wire format

**Newline-delimited JSON.** Each record is `json.dumps(obj) + "\n"`. LF only — strip a trailing `\r` before emit if present (pi-mono does this explicitly to handle Windows clients writing CRLF).

**No length-prefixing, no handshake.** The process spawns, immediately reads stdin, and writes events to stdout. Stderr is for human-readable diagnostics (logging) and is not part of the protocol.

**No backpressure.** Clients must drain stdout fast or buffer it. Phase 2 doesn't add flow control.

## Shape conventions

Pi-mono's pattern is two distinct outbound categories:

1. **Responses** to inbound commands (correlated by an optional `id` echoed from the request):
   ```json
   {"id": "req_1", "type": "response", "command": "prompt", "success": true}
   {"id": "req_2", "type": "response", "command": "get_messages", "success": true, "data": [...]}
   {"id": "req_3", "type": "response", "command": "prompt", "success": false, "error": "no model configured"}
   ```

2. **Async events** that flow during a `prompt`'s lifetime:
   ```json
   {"type": "user_message", "content": "..."}
   {"type": "assistant_text_delta", "delta": "Hel"}
   {"type": "tool_call_start", "id": "call_1", "name": "read"}
   {"type": "tool_call_delta", "id": "call_1", "delta": "{\"path\":\""}
   {"type": "tool_call_end", "id": "call_1", "name": "read", "arguments": {"path": "x"}}
   {"type": "tool_result", "tool_call_id": "call_1", "content": "...", "is_error": false}
   {"type": "agent_end"}
   ```

We mirror this **two-category** pattern. It keeps responses idempotent (one per command) while allowing events to flow between them.

## Inbound: minimum viable subset

```json
// Submit a message; turns into a streaming run.
{"id": "req_1", "type": "prompt", "message": "list files"}

// Cancel the current run (maps to asyncio task cancellation).
{"id": "req_2", "type": "abort"}

// Read history.
{"id": "req_3", "type": "get_messages"}
```

`id` is optional but if provided must be echoed on the response. Generate `req_<N>` if absent (matches `rpc-client.ts:479`).

`prompt` returns success **immediately** after preflight, before the run completes. Its `data.accepted` says whether the run `started` or was `queued` behind one already in flight — the response answers "did you accept this", never "did it finish".

Events follow asynchronously. **Clients wait on `agent_settled`, not `agent_end`.** `agent_end` means one run finished; a queued follow-up starts another, so it can fire several times for one client prompt. `agent_settled` is emitted once the run is done *and* the queues are drained, including on the error and abort paths.

## Outbound: event mapping from our internal taxonomy

Our internal `AgentEvent` (in `src/midge/agent.py` and `src/midge/client.py`) maps cleanly to pi-mono's wire events:

| Internal | Wire | Notes |
|---|---|---|
| `UserMessage` (appended in `Agent.stream`) | `{"type": "user_message", "content": <str-or-blocks>}` | Emit once at the start of each run |
| `StreamStart` | (skip) | Internal-only; nothing useful for clients |
| `TextStart` | (skip) | Implicit — first `text_delta` covers it |
| `TextDelta` | `{"type": "assistant_text_delta", "delta": "..."}` | |
| `TextEnd` | (skip) | |
| `ToolCallStart` | `{"type": "tool_call_start", "id": "...", "name": "..."}` | Pull id/name from the partial's `content[content_index]` |
| `ToolCallDelta` | `{"type": "tool_call_delta", "id": "...", "delta": "..."}` | |
| `ToolCallEnd` | `{"type": "tool_call_end", "id": "...", "name": "...", "arguments": {...}}` | |
| `Done` | `{"type": "assistant_message_end", "stop_reason": "...", "model": "..."}` | Optional, but nice for clients that want a per-LLM-call boundary |
| `Error` | `{"type": "error", "message": "...", "stop_reason": "error"\|"aborted"}` | |
| `ToolExecutionStart` | `{"type": "tool_execution_start", "id": "...", "name": "..."}` | Distinct from `tool_call_start` — the latter is the LLM emitting the call, this is us *running* it |
| `ToolExecutionEnd` | `{"type": "tool_result", "tool_call_id": "...", "content": "...", "is_error": false}` | |
| `AgentEnd` | `{"type": "agent_end"}` | Terminal event for the run |

The naming difference (`tool_call_*` for LLM emission vs. `tool_execution_*`/`tool_result` for execution) is **load-bearing**. External clients use the first to render "the model is requesting…" and the second to render "we ran it and got back…". Don't conflate them.

## Lifecycle

1. Process spawns with `midge --rpc` (or however we wire it; an `examples/rpc_agent.py` is the simplest start).
2. Read stdin line by line; parse each as a command.
3. Write responses (correlated by `id`) and events (uncorrelated) to stdout, each as one `json.dumps + "\n"`.
4. On stdin EOF, shut down cleanly.
5. SIGTERM/SIGHUP → cancel any in-flight task, exit. SIGKILL is brute-force only.

Implemented as `rpc.serve_stdio(server)`, which claims stdout, installs the
signal handlers and reads stdin to EOF. Both `midge --rpc` and
`examples/rpc_agent.py` go through it.

**Stdout is claimed at startup.** `claim_stdout()` captures the real handle for
the protocol and repoints `sys.stdout` at stderr, so a stray `print()` from a
tool, a hook, an extension or a dependency cannot corrupt the stream. This is
not hypothetical — an extension that printed on every hook event put three
non-JSON lines mid-stream before the guard existed.

## Errors

- **Parse error** on inbound: emit `{"type": "response", "command": "parse", "success": false, "error": "..."}`. Don't include an `id` because we couldn't parse it.
- **Unknown command type:** `{"type": "response", "command": <type>, "success": false, "error": "unknown command", "id": <if-present>}`.
- **Validation error inside a command** (e.g. `prompt` without `message`): same shape, with the offending field in the message.

## Subtle points

- **Stream interleaving.** When two commands arrive close together, the second only fires *after* the first's response is written. Pi-mono enforces this by serializing on a queue (`rpc-mode.ts`); we should do the same with a single asyncio task driving the stdin reader.
- **Async commands and events.** While a `prompt` is streaming, async events flow on stdout. A `get_messages` sent mid-prompt is answered *immediately*, interleaved with the event stream — the dispatch loop keeps reading stdin during a run so `abort` can land, and every other command rides the same path. (An earlier draft of this note said such commands were deferred until the run finished; that was never what the code did.)
- **No image support in v1.** `prompt` accepts `message: str` only. `images` is deferred.
- **JSON encoding gotcha.** `json.dumps` defaults to ensure_ascii=True; turn it off (`ensure_ascii=False`) so unicode survives without `\u` escapes — keeps wire payloads small and readable.

## What Phase 2 implements

- `src/midge/rpc.py`:
  - `RpcServer.serve(read_line=, write=)` — the transport-agnostic loop; the injected callables are what make it testable without pipes.
  - `serve_stdio(server)` — binds that loop to this process's stdin/stdout.
  - One handler per inbound command type, dispatched by a `match` on `type`. (This note originally said "use a dispatch dict, not isinstance chains". A `match` statement reads better than either and is what shipped.)
  - `event_to_wire(ev) -> dict | None` — returns None for events we drop. Named `agent_event_to_wire` in this note originally.
- `examples/rpc_agent.py`:
  - Tiny entrypoint mirroring `examples/coding_agent.py`'s wiring, ending in `await serve_stdio(server)`. `midge --rpc` is the first-class route and gets `--session`, `--skill-dir` and the compaction flags; the example stays minimal on purpose.
- Tests in `tests/test_rpc.py`:
  - Drive `serve()` with a queue-backed fake stdin and a buffer-backed fake stdout, asserting on parsed frames. Keeping the transport injectable is what lets the whole protocol be tested in-process.

**Keep the event-mapping seam.** pi pipes its internal session events straight to stdout, which is free for pi but couples every client to its internal types — that is how TUI-only events ended up on its wire. Add new events through `event_to_wire`, and keep the explicit `tool_call_*` (the model is requesting) vs `tool_execution_*` / `tool_result` (we ran it) split; in pi that distinction is implied by ordering inside a nested payload.
