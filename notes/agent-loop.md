# Agent loop — patterns to borrow from `pi-mono`

Source:
- `pi-mono/packages/agent/src/agent-loop.ts` — the actual loop (state machine)
- `pi-mono/packages/agent/src/agent.ts` — stateful wrapper, queues, event emission
- `pi-mono/packages/coding-agent/src/core/agent-session.ts` — integration with persistence, compaction, extensions

> **Important finding:** the loop lives in the **`agent`** package (generic), not the **`coding-agent`** package. The coding-agent only wraps it. This separation is exactly the pattern we want to keep — the harness loop should not know it's a coding agent.

## Two-level loop, but Phase 1 only needs the inner one

The TS loop is two layers:

1. **Outer loop** — handles "follow-up messages" injected after the agent would naturally stop. Only triggers if the inner loop ran out of work.
2. **Inner loop** — the per-turn cycle: send → stream → execute tools → loop until no tool calls remain.

For Phase 1 we implement the **inner loop only**. Steering and follow-up message queues are advanced features; skip them.

## Inner loop — order of operations

Each turn does this:

1. **Send.** Append the pending user/tool messages to history. Call the LLM with the full message list.
2. **Stream.** Iterate the response stream with `async for`. Maintain a single mutable "partial assistant message" that gets updated in place on each delta. UI consumers hold a reference to this object.
3. **Finalize.** On the `done` event, the partial message becomes the final assistant message. On `error`, finalize with `stop_reason="error"`.
4. **Dispatch tools.** If the assistant message contains tool calls, execute them. Append tool result messages to history.
5. **Loop or exit.** If there were tool calls, loop. Otherwise emit `agent_end` and exit.

## Termination conditions

The loop exits when one of these is true after a turn:

- The assistant message has `stop_reason in {"error", "aborted"}` — exit immediately, do not run tools.
- The assistant message has no tool calls — exit normally.
- (Phase 1: that's it. Skip steering/follow-up.)

There is no explicit max-turns limit. Compaction is what prevents runaway context, not a turn cap.

## Tool dispatch — sequential vs parallel

- Default is parallel. Each tool is a coroutine; collect them and `asyncio.gather()`.
- Per-tool override: a tool can declare `execution_mode="sequential"` and is then run alone.
- Emit a `tool_execution_start` event before each tool, `tool_execution_end` after.
- Tool errors (validation failure, raised exception, missing tool) become tool-result messages with `is_error=True`. The model sees them on the next turn.
- Results are matched to their tool calls by `tool_call_id` (round-tripped from the streaming events).

## Streaming consumption pattern

```
partial = AssistantMessage()  # mutable; held in history
history.append(partial)

async for event in client.stream(history, tools):
    match event.type:
        case "text_delta":   partial.content += event.delta
        case "toolcall_delta": <update partial.tool_calls[i].args_buffer>
        case "done":           final = event.message; replace partial in-place
        case "error":          finalize partial with stop_reason="error"; raise
```

The mutate-in-place pattern is **load-bearing** for streaming UIs — observers see the same object reference grow. Don't replace the partial with a new instance on every delta.

## Error handling

- **Stream error mid-stream:** the provider yields an `error` event with a finalized message (stop_reason="error"). Loop catches, exits cleanly.
- **Tool execution error:** caught at the dispatch site, converted to an error tool-result. Loop continues.
- **Abort signal:** propagates as `asyncio.CancelledError`. The loop should catch at its boundary and finalize with `stop_reason="aborted"`.
- **Outer catch:** if the executor itself throws unexpectedly, synthesize an assistant message with `stop_reason="error"` and append. Don't let the harness die.

## Non-obvious load-bearing details

1. **Mutable context with in-place message updates.** Streaming UIs depend on this.
2. **`new_messages` accumulator.** Track everything added during this run separately from the full history. Useful for retries, compaction, and the "what did this turn produce" question.
3. **`convert_to_llm` boundary.** History internally can hold custom message types (extensions, UI annotations). Filter to LLM-compatible messages only at the stream call. Phase 1 doesn't need custom types, but build the boundary so it's there.
4. **Steering polled via callback.** The TS loop calls `config.get_steering_messages()` between turns rather than reading a queue. This decouples queue management from loop logic. Phase 1 skips steering, but if we add it later, do it the same way.

## What Phase 1 implements

- Single async loop, no outer/inner split.
- Mutate-in-place partial message during streaming.
- Termination: error/abort or no tool calls.
- Parallel tool dispatch via `asyncio.gather`.
- Tool errors become error result messages.
- No compaction, no save/load, no steering, no follow-up, no extensions.

That's the minimum viable harness loop. ~150–200 lines of Python.
