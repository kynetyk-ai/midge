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
>
> **Update (#49).** `set_session_name` shipped, and `clear_context` became
> durable. The storage decision was that mutable state is an appended record
> replayed on load, never a header rewrite — see `notes/sessions.md`. Fork /
> clone / switch remain out; they need the entry tree, which markers do not.

## Wire format

**Newline-delimited JSON.** Each record is `json.dumps(obj) + "\n"`. LF only — strip a trailing `\r` before emit if present (pi-mono does this explicitly to handle Windows clients writing CRLF).

**No length-prefixing, no handshake.** The process spawns, immediately reads stdin, and writes events to stdout. Stderr is for human-readable diagnostics (logging) and is not part of the protocol.

**Backpressure.** A pipe holds ~64 KiB and one ordinary answer is ~20 KiB of frames — every token is its own record — so roughly three answers fill it, and one `read` of a large file produces a single `tool_result` frame near 50 KiB. A client that stops draining is therefore normal, not exotic.

Two things keep that from wedging the process:

- **The writer drains rather than blocks.** `loop.connect_write_pipe` plus `StreamWriter.drain()`, so a full pipe suspends the writing coroutine instead of the event loop. A plain `file.write` blocks the loop, which stops the agent, every tool, *and* the stdin reader together — `abort` cannot arrive because it is read by the blocked loop, and the SIGTERM handler cannot run because it is queued on it. Only SIGKILL is left. Regular files fall back to blocking writes: asyncio refuses to wrap them, and a file has no reader to stall behind.
- **Frames go through a bounded outbox** drained by one writer task, so the dispatch loop never awaits a write. Without that, a stalled write inside a command handler stops the loop from reading the next command — including `abort`.

The queue is `OUTBOX_FRAMES` deep, about ten answers' worth: deep enough that a client pausing to render or collect garbage never stalls anything, shallow enough that a client which has genuinely died applies backpressure rather than exhausting memory. When it fills, the producer waits — which is the agent, since it emits nearly all the frames. That matches pi, which stalls its agent loop for the same reason: pausing beats dropping protocol frames or buffering without bound.

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

## Steering, follow-up, and the terminal event

`steer {message}` is delivered at a **tool-call boundary inside the run already
going** — after every tool result for the current turn is appended, before the
next provider request is built. "Interrupt at the next safe seam", not
"interrupt now": a steer issued during a ten-tool batch waits for that batch.
It also re-arms a turn that answered in plain text, so a steer never sits
stranded until the next prompt.

`follow_up {message}` is delivered only once the run has nothing left to do,
which makes it the next turn. A `prompt` arriving mid-run is queued as a
follow-up rather than refused; the response says `data.accepted` = `started` or
`queued` so a client never has to infer which.

Ordering between the two queues is **priority, not arrival**: steering drains at
every boundary, follow-up only at quiescence, so a stream of steers delays a
follow-up indefinitely even if it was queued first.

**There is exactly one safe place to inject.** Providers reject a request where
a `tool` message does not follow the assistant message that issued its call, and
`to_openai_messages` is a single forward pass that will not repair a split. The
loop edge is the only point where every result is in and the next request has
not been built.

**Abort clears the queues** and returns what it dropped, so a client can put the
text back in front of the user. pi leaves its queues alone, which means aborting
a turn silently starts a *new* run from whatever was pending — every pi UI works
around that, so the workaround belongs in the core.

`queue_update` carries a full snapshot, `{steering: [...], follow_up: [...]}`,
on every enqueue and drain. Entries carry ids; matching on text is ambiguous for
duplicates and blind to anything that is not text.

**Whatever is queued must already be a plain message.** Anything whose meaning
depends on when it fires has to be rejected or resolved at enqueue time, so its
errors reach whoever queued it rather than surfacing mid-run with nothing to
attribute them to. midge has no command layer yet; the rule is written down so
the future one obeys it.

## Enumerating commands

`get_commands` answers "what can a user invoke, and how", and executes nothing.
It is a **projection, not an invention**: built-ins come from the dispatch
table, skills from directories on disk. That is what makes it safe to ship
before a consumer exists — the failure mode is awkward field names, not a wrong
feature, and nothing depends on it yet.

Each entry carries four things:

- `invoke` — how to transmit. `command` means send `{"type": name, …}`;
  `prompt` means put text in a `prompt`/`steer`/`follow_up` message.
- `parameters` — JSON Schema, the same shape `Tool.schema()` produces. **Empty
  `properties` is the select-and-fire signal**, so a consumer learns arity from
  the data rather than a separate flag. A client can build a form from this and
  drive a command it knows nothing else about.
- `description` — the label.
- `source_info` — provenance; skills only, since they load from four default
  directories plus `--skill-dir` and "which one is this?" is a real question.

`parameters` means slightly different things per `invoke`: for a command the
properties are keys in the request object; for a prompt the single property is
free text appended after the name (`/skill:deploy to staging`). A
prompt-invoked command takes at most one argument, which is what keeps that
unambiguous — a source needing more wants a real answer, not a cleverer reading
of this one.

**Deliberately absent: any notion of danger.** `clear_context`, `new_session`
and `abort` all discard or interrupt something, but whether that warrants a
confirmation is consumer policy — a misclick in a terminal and one in a shared
channel are different risks, and the server does not know which it is talking
to. No UI concepts in the response.

**What is listed.** The server has an opinion about what is a user-facing
action: `abort`, `compact`, `clear_context`, `new_session`, `export_html`,
`set_model`, `set_system_prompt`, `reload`, `set_session_name`, plus every
loaded skill as `skill:<name>`.
Out: `prompt`, `steer` and `follow_up`, which *are* the interaction rather than
menu items, and the `get_*` family, which a client reads to render itself.

Skills are listed regardless of `disable-model-invocation` — hiding one from the
model's catalogue is exactly the case where an explicit command is the only way
to reach it.

`/skill:name [args]` is expanded server-side into the `<skill …>` envelope, at
**enqueue** time for `steer` and `follow_up`. That is the rule stated above: a
bad name then fails in the response to whoever queued it rather than surfacing
mid-run, and what gets delivered is frozen at the moment the user asked for it.
An unknown name is an error, not literal text — pi passes it through because it
has a text-expansion path where that makes sense; midge does not.

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

## `reload` (#46)

Skills and extensions were discovered once, at startup, so a new `SKILL.md` or an
edited extension needed a restart to take effect. That is a bad loop for
authoring and a worse one under RPC, where the process is long-lived by design.
`pi` has `/reload` for the same reason (`slash-commands.ts:40`).

**Hooks are not a third target.** There is no `--hook-dir` and no hook file
format; hooks reach the registry only through `load_extensions`, which calls each
module's `register_hooks`. Reloading extensions *is* reloading hooks.

### Both targets are a discard and re-run

Nothing incremental, nothing source-scoped — the loaders already have the shapes
that make the blunt version correct:

- **Tools are returned.** `load_extensions` builds a *fresh* `ToolRegistry`, so
  assigning it drops the old one whole. No removal logic, no residue.
- **Hooks are mutated** into a shared `Hooks` rather than returned, so they would
  double-register. `Hooks.clear()` handles it, and it awaits every `add_cleanup`
  handler first — exactly what unloading should do.
- **Sub-agents rebind for free.** Re-import produces new `SubagentTool`
  instances, so `bind_subagents` on the new registry binds them. The model comes
  off the agent, so a `set_model` since startup is respected rather than reverted.

A source-scoped hook removal was designed and dropped. It would do identical work
today, because `load_extensions` is the only thing that registers into the
server's `Hooks`. `_Registration.source` is already stamped if that ever stops
being true — that is the upgrade path, and it is not needed yet.

### The server stores source lists, not a recipe

`extension_sources` and `skill_sources` are the exact lists the entrypoint passed
to the loaders. Reload repeats that call. Reconstructing them server-side would
mean knowing which sources are built-in, and an embedder that handed the agent a
deliberately restricted registry would find reload silently widening it to every
built-in tool. `None` means the entrypoint did not wire that target up, which is
not an empty list: naming an unwired target is an error, while the bare form
reloads whatever *is* wired, so the convenient spelling always works.

### Refused mid-turn

Swapping the tool registry under a running turn breaks tool-call/result pairing.
Refusing also disposes of the one genuinely hard case: a child registry bound to
the old `SubagentRuntime` can only exist inside an in-flight sub-agent, which can
only exist inside a turn.

### The one coupling: the read gate

The skills catalogue tells the model to open a `SKILL.md`, so it is gated on a
`read` tool — without one it is an instruction to do the impossible. That gate is
*derived* in `_generated_prompt()` rather than stored, so it cannot fall out of
date. The consequence is that reloading **extensions** can add or remove the
skills catalogue with no skill having changed, which is the only place the two
targets are not independent.

`_base_prompt` is untouched, so a prompt set via `set_system_prompt` survives.

Since #57 it survives a *resume* too: `set_system_prompt` and `set_model` append
an `identity` / `model_change` record to the session, and `cli.py` folds them
back through `resume_identity`. Before that both were process-lifetime only and
reverted silently. Persistence stays optional — without a session the commands
still succeed, and the response's `durable` is what says which happened.

### Not on the wire

Per-file errors. Both loaders already skip a bad file, log it, and carry on;
reporting them here means changing both return types to serve a client that does
not exist. The response carries the resulting `tools` and `skills` counts.

### Still deferred

A re-imported extension gets a fresh module under a new synthetic name, so the
old module's state is not reclaimed. `notes/extensions.md` describes extensions
as stateless; reload is what makes that assumption matter.
