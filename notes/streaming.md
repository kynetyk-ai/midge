# Streaming + LLM client — patterns to borrow from `pi-mono`

Source:
- `pi-mono/packages/ai/src/utils/event-stream.ts` — generic AsyncIterable wrapper
- `pi-mono/packages/ai/src/types.ts` — `AssistantMessageEvent` union (lines 259–271)
- `pi-mono/packages/ai/src/providers/openai-responses.ts` — provider entrypoint
- `pi-mono/packages/ai/src/providers/openai-responses-shared.ts` — `processResponsesStream` and message conversion
- `pi-mono/packages/ai/src/utils/json-parse.ts` — `parseStreamingJson`

## Important deviation: Chat Completions, not Responses API

`pi-mono` targets OpenAI's **Responses API** (the newer one, with `client.responses.create`). For our port, we should use **Chat Completions** (`client.chat.completions.create`) instead, because:

- Local/self-hosted OpenAI-compatible servers (ollama, vLLM, LM Studio, llama.cpp's `server`, OpenRouter, Together, Groq) almost universally implement Chat Completions, not Responses.
- Our roadmap explicitly targets these as supported backends.
- Chat Completions streaming is simpler and well-documented.

The event taxonomy below still applies — we just map from a different wire format. The patterns transfer cleanly.

## Event taxonomy (the harness's internal events)

The harness exposes a single async generator that yields these event types. We should match the names so external clients are portable:

| Event | When |
|---|---|
| `start` | Stream opens, before any content |
| `text_start` / `text_delta` / `text_end` | Assistant text content lifecycle |
| `thinking_start` / `thinking_delta` / `thinking_end` | Reasoning blocks (only models that emit them) |
| `toolcall_start` / `toolcall_delta` / `toolcall_end` | Tool call construction; `_delta` carries argument JSON chunks |
| `done` | Terminal success — carries the final `AssistantMessage` |
| `error` | Terminal failure — carries an `AssistantMessage` with `stop_reason="error"` or `"aborted"` |

Each content event includes a `content_index` (position in the message's content array) so multiple parallel tool calls can be tracked independently.

For Phase 1 we may collapse `thinking_*` events out (most OpenAI-compat models don't emit reasoning) and add them back when needed.

## Partial-JSON tool argument assembly

This is the fiddly part. Tool arguments stream in as JSON-string chunks (`{"path"`, ` "/etc"`, `, "limit": 10}`). To give the harness incrementally usable arguments before the tool call completes, the TS code parses **on every delta** with a four-tier fallback:

1. Try strict `JSON.parse(buffer)`.
2. If that fails, try `partial-json` library (understands unclosed structures).
3. If that fails, run a small "repair" pass and try `partial-json` again.
4. Last resort: empty object `{}`.

At the final `arguments.done` event, the API hands us the complete valid JSON; parse it once authoritatively and replace the buffered version. Delete the temporary buffer field after `toolcall_end` so it doesn't leak into history.

**Python equivalents for partial parsing:**
- `jiter` (already a transitive dep via `openai`) supports `partial_mode="trailing-strings"` — this is probably the cleanest answer.
- Or stdlib `json.loads` with try/except, fall back to "wait for done event" — simpler, slightly worse UX (UI sees no progress on the args until they're complete).

Phase 1 can start with the simpler approach (`json.loads` on `done` only) and upgrade to incremental parsing later if the UX needs it.

## OpenAI Chat Completions → harness events

Chat Completions streaming yields chunks shaped like:

```
{
  choices: [{
    delta: {
      content?: str,
      tool_calls?: [{
        index: int,
        id?: str,
        function?: { name?: str, arguments?: str }  // arguments are STREAMED as a string
      }]
    },
    finish_reason?: "stop" | "tool_calls" | "length" | "content_filter"
  }]
}
```

Mapping to our events:

- `delta.content` non-empty → `text_delta` (with `text_start` on first occurrence, `text_end` when content stops or message ends).
- `delta.tool_calls[i]` first appearance for that index → `toolcall_start` (capture id + function name).
- `delta.tool_calls[i].function.arguments` → append to the buffer for index `i`, emit `toolcall_delta`.
- `finish_reason` non-null on the chunk → finalize: emit `_end` for any open block, emit `done`.

`finish_reason` mapping: `"stop"` → `stop`, `"tool_calls"` → `toolUse`, `"length"` → `length`, `"content_filter"` → `error`.

**Multi-tool-call quirk:** OpenAI emits multiple tool calls by interleaving deltas with different `index` values. Maintain a dict keyed by index, not a single accumulator.

## Tool result format on the way back

For Chat Completions, a tool result message looks like:

```python
{"role": "tool", "tool_call_id": "<id>", "content": "<text result>"}
```

If the tool returns images and the model supports vision, content becomes a list of content parts (text + image_url). For Phase 1, text-only is fine — coding tools rarely produce images directly.

## Errors and aborts

- **Mid-stream HTTP/network error:** the OpenAI client raises. Catch in the stream loop, emit `error` event, finalize.
- **AbortSignal / cancellation:** Python equivalent is `asyncio.CancelledError`. Pass an `asyncio.Event` or use `asyncio.timeout()` for cancellation; on cancel, emit an `error` event with `stop_reason="aborted"` and re-raise.

## Subtle behaviors worth knowing

- **Partial JSON with valid prefix:** `{"path": "/etc"` is partially valid (key is complete, value is a complete string). `jiter` with `partial_mode` returns `{"path": "/etc"}` from this. This is what makes streaming arg display work.
- **Tool calls can split across many chunks.** Don't assume one chunk = one tool call.
- **`finish_reason` arrives on the chunk where the message ends, not on a separate "stream end" event.** When you see it, the next chunk is the close of the iterator.
- **Reasoning/thinking content from o1/o3-style models** is delivered in a different field (`reasoning_content` on Chat Completions for some providers, or via specific event types on Responses). Skip this in Phase 1 unless we explicitly target reasoning models.
- **System fingerprint, usage tokens, and service tier** appear on the final chunk — and critically, that chunk's `choices` array is **empty**, so the usual `if not chunk.choices: continue` guard at the top of a chunk loop is exactly what discards it. Usage must be read before that guard.

  `usage` is now captured onto `AssistantMessage.usage` (`client.py`), requested via `stream_options={"include_usage": True}` behind `MIDGE_INCLUDE_USAGE` because support varies across OpenAI-compatible servers and a server that rejects it fails the whole turn. `compaction.measure_context` prefers it over the bytes/4 estimator, which is the point: compaction destroys history and should not trigger on an inflated proxy.

  Deliberately **not** captured: system fingerprint and service tier (nothing reads them), and no cost — that needs a price table, which goes stale silently, and cost is a fold over the session file with whatever prices the reader trusts at the time they ask.

## Phase 1 plan for our client

A minimal `client.py`:

- Wrap `openai.AsyncOpenAI` (configurable `api_key` and `base_url`).
- Single method: `stream(messages, tools, model, **kwargs) -> AsyncIterator[StreamEvent]`.
- Internally: call `chat.completions.create(stream=True)`, iterate chunks, dispatch to our event taxonomy.
- Buffer tool-call arguments; emit `toolcall_delta` with the raw chunk; on `_end`, parse the complete JSON via `json.loads`.
- Skip incremental partial-JSON parsing for v1; add `jiter` partial mode in v2 if we want progressively-displayed tool args.

## Update — the provider seam

The chunk-parsing described above now lives in `src/midge/providers/openai_compat.py`
rather than in `client.py`. The split is:

- **`providers/`** owns one wire format each: `encode` builds a request body,
  `open` starts it, `decode` turns one vendor chunk into a `Delta`, and
  `is_retryable` classifies a failure. Nothing else.
- **`client.py`** owns the part that is the same whatever you talk to: the
  streaming state machine (partial assembly, content indices, tool-argument
  buffering) and the retry policy.

`Delta` is the normalization point — text fragment, tool-call fragments, a
`StopReason`, and `Usage`. Above it, midge's vocabulary; below it, the vendor's.

**The state machine deliberately does not move into the adapter.** It is the
subtle part — the "usage rides a final chunk whose `choices` is empty" quirk, the
invariant that a retry only happens before any content event escaped, index
bookkeeping across interleaved parallel tool calls — and a second copy per vendor
would be the main source of bugs. One state machine, N translators.

Two names are registered against the same adapter, `openai` and
`openai-compatible`, because they share this wire format entirely. What differs
is what a server tolerates, and that is declared as `Capabilities`
(`requires_api_key`, `stream_usage`) rather than discovered by catching a 400.
`MIDGE_INCLUDE_USAGE` survives as an override for servers that reject
`stream_options`, but the default now comes from the provider.

The note above also predates a related split: `to_openai_messages` became
`messages.repair_history` (dropping failed assistant turns and orphaned tool
results — a fact about midge's history, so provider-independent) plus the
adapter's `encode_messages`. Core repairs once; each provider only encodes.
