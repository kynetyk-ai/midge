# Agent loop

`Agent` in [`src/midge/agent.py`](../src/midge/agent.py) runs a turn: it sends the history to the model, streams the reply, executes any tool calls, and repeats until the model answers without calling a tool. It knows nothing about coding; tools, prompt and model are all passed in.

## Messages

History is a list of three Pydantic models from [`src/midge/messages.py`](../src/midge/messages.py):

| Message | Holds |
|---|---|
| `UserMessage` | A string, or a list of text and image blocks |
| `AssistantMessage` | Text and `ToolCall` blocks, plus `model`, `stop_reason`, `error_message`, `usage` and `extra` |
| `ToolResultMessage` | The result for one `tool_call_id`, with `is_error` |

`stop_reason` is one of `stop`, `length`, `tool_use`, `error` or `aborted`. `usage` holds the token counts the provider reported (`input`, `output`, `cached`). `extra` carries provider state that has to round-trip with history; the loop stores it without reading it (see [providers](providers.md)).

## One turn

`Agent.stream(user_input)` is an async generator. Each iteration of its loop:

1. Builds the request from `history`, the model, the tool schemas and the system prompt. `context` and `before_provider_request` hooks can replace any of these for this request only.
2. Streams the reply through `Client.stream` and yields every stream event. The partial `AssistantMessage` handed out by the `start` event is the same object that grows as deltas arrive, so a UI can hold that reference.
3. Appends the finished assistant message to history, after the `after_provider_response` hook may replace it.
4. Stops if `stop_reason` is `error` or `aborted`, or if the message has no tool calls.
5. Otherwise runs the tool calls, appends one `ToolResultMessage` per call, and loops.

`turn_start` fires before the first request and can replace the system prompt or prepend messages; `turn_end` fires once at the end. The hook events and their result types are described in [hooks](hooks.md).

There is no turn limit. Long runs are bounded by [compaction](compaction.md), which runs between turns.

## Tool calls

Every tool call ends up with a result in history, because providers reject a request that contains a tool call with no matching result.

- **Ordering.** Consecutive calls to tools declared `read_only` run concurrently. Any other call waits for everything before it and runs alone, so a model that writes a file and then edits it in one message gets the edit applied to the written file. See [tools](tools.md) for `read_only`.
- **Hooks.** `tool_call` hooks run for every call before any executes. A hook can rewrite a call's arguments or block it; a blocked call gets an error result carrying the hook's reason. `tool_result` hooks can then rewrite each result's content or `is_error`.
- **Errors.** An unknown tool, invalid arguments, arguments the provider stream could not parse, or an exception raised by the tool become a result with `is_error=True`. The model sees it on the next request.
- **Truncated replies.** When `stop_reason` is `length`, the tool calls may be incomplete, so none run. Each gets an error result asking the model to re-issue it, and the turn ends.
- **Result size.** A result longer than `MAX_TOOL_RESULT_CHARS` is cut and marked, because extension tools do not bound their own output and every result is resent on later requests.
- **Interruption.** If the turn is cancelled while tools run, calls that already finished keep their real results and the rest get an "interrupted" error result before the cancellation propagates.

## History repair

History records failed turns as they happened. Before each request, `messages.repair_history` removes what no provider accepts: assistant messages that errored, were aborted, or said nothing, and tool results whose call is no longer present. The stored history is not modified. This is the single normalization point for damage from the loop, a hook, compaction or a resumed session.

## Steering and follow-up

`SteeringQueue` lets a client add messages to a run that is already going. Entries get an id (`q1`, `q2`, ...) so a client can match a delivery to what it queued.

- **Steering** messages are appended at the loop edge: before the first request, and after every tool result for the turn is in history. A steer that arrives during a text-only reply starts another iteration instead of waiting for the next prompt. Each injection yields a `steered` event.
- **Follow-up** messages are not read by the loop. A front-end takes one after a run ends and starts a new run with it; the RPC server does this (see [rpc](rpc.md)).
- `clear()` drops both queues and returns what was dropped. Aborting a run clears them, so a cancelled run does not restart with stale input.

## Events

`Agent.stream` yields the provider stream events ([providers](providers.md)) plus:

| Event | When |
|---|---|
| `tool_execution_start` | Before a tool call runs (also for blocked and truncated calls) |
| `tool_execution_end` | With the call's final `ToolResultMessage` |
| `steered` | A queued steering message was appended to history |
| `agent_end` | The run is over; carries `new_messages`, everything this run added |

## Using it

- `Agent.stream` mutates `history` in place and refuses to start while another stream is running. Cancel and await the current one before starting another.
- `Agent.run` drives `stream` to completion and returns the last assistant message.
- `model`, `system_prompt`, `tools` and `history` are plain attributes. The model is read at every request, so changing it takes effect on the next one.
- Both front-ends call `Controls.run_turn` in [`src/midge/commands.py`](../src/midge/commands.py) instead of `Agent.stream` directly. It adds persistence ([sessions](sessions.md)) and automatic [compaction](compaction.md) around the loop.

[`examples/coding_agent.py`](../examples/coding_agent.py) wires an `Agent`, a `Client`, a session and compaction by hand in one file.
