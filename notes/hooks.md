# Hooks — lifecycle interception

Source: `pi-mono/packages/agent/docs/hooks.md` (445 lines, marked "Final design").

Ported to `src/pym/hooks.py`. Concepts follow pi; the Python is written from scratch.

## The core split

- `observe(handler)` — sees every event, read-only, return ignored.
- `on(type, handler)` — participates in that event's semantics.
- `emit(event)` — the only thing the loop calls.

The loop never stores handlers or knows extension policy; it emits and applies the result.

## Why per-event reduction rules

The non-obvious part is that "what do I do with N handlers' return values" differs per event,
and getting it wrong is silent. `Hooks._REDUCERS` maps each event type to one of five rules:

| Rule | Events | Behavior |
|---|---|---|
| observation | `message_end`, `turn_end` | run all, ignore returns |
| first-cancel-or-last | `session_start`, `session_end`, `before_compact` | stop at the first `cancel=True` |
| chain | `context`, `after_provider_response` | each handler sees the previous handler's output |
| accumulate | `before_provider_request`, `tool_result` | patches merge field-by-field |
| early-exit-on-block | `tool_call` | argument rewrites chain; the first `block=True` wins and stops |

Adding an event without adding a reducer makes it silently observational. That is pi's
"poking holes" item #6, and it applies here identically — `_REDUCERS` is the one place to look.

## Departures from pi

**No phantom types.** pi encodes each event's result type as a TS type-only symbol. Python has no
equivalent, so `on()` carries an `@overload` per event instead. Same checking, different mechanism,
and a plain `on(type, handler)` fallback overload keeps custom event types open to extensions.

**`after_provider_response` vs `message_end`.** In pi these are separated by a provider-payload
layer that py-mono doesn't have, so both would fire on the same object at the same instant.
Given distinct semantics here instead: `after_provider_response` is a **transform** that can
replace the `AssistantMessage` before it enters `history`; `message_end` is **observe-only**,
after the append.

**`Hooks` is not owned by `Agent`.** `before_compact` fires inside `compaction.compact()` and the
session events fire wherever the `Session` lives — neither is inside the agent loop. The
entrypoint constructs `Hooks` and passes it down. `tui/app.py` reaches `compact()` via
`self.agent.hooks` rather than taking a separate parameter.

**No `AbortSignal`.** pi threads one through every handler. Python has task cancellation, so
handlers just get `(event, context)`.

## The concurrency constraint

`Agent` runs tools concurrently via `asyncio.gather`. `tool_call` hooks therefore have to resolve
**before** the gather, or a blocked call would already have run. The loop:

1. emits `tool_call` for every call (concurrent across calls, sequential within one)
2. applies any argument rewrites to `tool_calls`
3. gathers only the un-blocked calls
4. scatters results back into their **original index** and synthesizes `_tool_error` for blocked ones

Step 4 is load-bearing: `zip(tool_calls, results, strict=True)` downstream assumes positional
correspondence. `tests/test_hooks.py::test_ordering_preserved_when_some_calls_blocked` covers
exactly this and fails if the scatter is replaced with a naive `enumerate`.

Blocked calls still yield `ToolExecutionStart`/`ToolExecutionEnd` so the TUI renders them as
errors rather than appearing to hang.

## Error policy

Default `error_mode="continue"`: a raising handler is logged at WARNING with its `source` and
skipped. This matches how `load_extensions` already treats a failing extension file.
`error_mode="raise"` exists for tests and strict embedding. pi flags explicit error policy as the
first thing a hook system needs, and it is.

## Extension integration

An extension defines `register_hooks(hooks)`; `load_extensions(..., hooks=...)` calls it with a
source-tagged proxy so handler failures name the file. See `examples/approval_extension/`.

## Not ported

- Exposing hooks over RPC
- Hot-reload / `set_hooks()` on a running agent (pi defers this too)
- Registries that aren't hooks — commands, renderers, providers. pi keeps these separate
  deliberately (its "poking holes" item #3); py-mono has no equivalent surface.
