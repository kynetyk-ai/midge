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

`prompt` returns success **immediately** after preflight, before the run completes. Events follow asynchronously, terminating with either `agent_end` or `error`. Clients track lifecycle by watching for those terminal events, not by waiting on the response.

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
5. SIGTERM → cancel any in-flight task, drain pending events, exit. SIGKILL is brute-force only.

## Errors

- **Parse error** on inbound: emit `{"type": "response", "command": "parse", "success": false, "error": "..."}`. Don't include an `id` because we couldn't parse it.
- **Unknown command type:** `{"type": "response", "command": <type>, "success": false, "error": "unknown command", "id": <if-present>}`.
- **Validation error inside a command** (e.g. `prompt` without `message`): same shape, with the offending field in the message.

## Subtle points

- **Stream interleaving.** When two commands arrive close together, the second only fires *after* the first's response is written. Pi-mono enforces this by serializing on a queue (`rpc-mode.ts`); we should do the same with a single asyncio task driving the stdin reader.
- **Async commands and events.** While a `prompt` is streaming, async events flow on stdout. If the client sends a `get_messages` mid-prompt, our serialized loop will run it after the prompt completes (or after `abort` interrupts it). Document this.
- **No image support in v1.** `prompt` accepts `message: str` only. `images` is deferred.
- **JSON encoding gotcha.** `json.dumps` defaults to ensure_ascii=True; turn it off (`ensure_ascii=False`) so unicode survives without `\u` escapes — keeps wire payloads small and readable.

## What Phase 2 implements

- `src/midge/rpc.py`:
  - A `serve_stdio()` async function that reads stdin, dispatches commands, writes events.
  - One handler per inbound command type. Use a dispatch dict, not isinstance chains.
  - Internal mapper `agent_event_to_wire(ev) -> dict | None` — returns None for events we drop.
- `examples/rpc_agent.py`:
  - Tiny entrypoint mirroring `examples/coding_agent.py`'s wiring, but ending in `await serve_stdio(agent)` instead of running a single prompt.
- Tests in `tests/test_rpc.py`:
  - Drive `serve_stdio` with `asyncio.StreamReader`/`StreamWriter` pairs (or `io.BytesIO` wrappers) and assert event sequences.

~150 lines for the RPC layer plus tests.
