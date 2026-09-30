# Hooks

Hooks let code run at fixed points in the agent's lifecycle and change what happens there: block or rewrite a tool call, patch a result, reshape the request sent to the model, or veto a compaction. The registry and event types are in [`src/midge/hooks.py`](../src/midge/hooks.py); most events are emitted from [`src/midge/agent.py`](../src/midge/agent.py).

## Registering handlers

An [extension](extensions.md) registers handlers by defining `register_hooks`:

```python
from midge.hooks import ToolCallResult

def register_hooks(hooks):
    hooks.on("tool_call", guard)
    hooks.observe(audit)
    hooks.add_cleanup(close_connection)

def guard(event, ctx):
    if event.tool_call.name == "bash":
        return ToolCallResult(block=True, reason="No shell in this domain.")
    return None
```

- `on(type, handler)` subscribes to one event type. The handler's return value takes part in that event, as described below. Returning `None` means no opinion.
- `observe(handler)` sees every event before the `on` handlers run. Its return value is ignored.
- `add_cleanup(fn)` registers a function to run when extensions are unloaded (see [Cleanup](#cleanup)).

`on` and `observe` return a function that unsubscribes the handler. Handlers are called as `handler(event, context)` and may be sync or async. `context` is whatever was passed to `Hooks(context)` or `set_context`; the `midge` entrypoint leaves it `None`. Events are frozen dataclasses; a handler changes behaviour only through its return value.

A handler that raises is logged as `hook_handler_failed` with its source and traceback, and treated as having returned `None`. `Hooks(error_mode="raise")` propagates the exception instead, for tests and strict embedding.

## Events

Handlers for one event run in registration order. The last column says how their results combine.

| Event | When | A handler may return | Combination |
|---|---|---|---|
| `session_start` | Process start, in both TUI and RPC mode, before the first prompt. Carries the session `path`. | `CancelResult(cancel=True)` to refuse startup: midge runs cleanups, logs `startup_cancelled_by_hook` and exits non-zero. | Stops at the first `cancel=True`. |
| `session_end` | Process exit, in both modes. Carries the session `path`. | Nothing. | Observation only. |
| `turn_start` | A prompt arrives, before it enters history. | `TurnStartResult(messages=, system_prompt=)`: messages to insert ahead of the prompt, and a system prompt for this turn only. | Messages accumulate; each handler sees the previous handler's system prompt. |
| `context` | Before each model request. | `ContextResult(messages=)`: the messages to send for this request. History is unchanged. | Chained. |
| `before_provider_request` | Before each model request, after `context`. | `ProviderRequestResult(model=, system=, tools=, kwargs=)`. | Each non-`None` field replaces the current value; the next handler sees the result. |
| `after_provider_response` | A model response is complete, before it enters history. | `ProviderResponseResult(message=)`: a replacement `AssistantMessage`. | Chained. |
| `message_end` | After the response is appended to history. | Nothing. | Observation only. |
| `tool_call` | Once per tool call, before any call in the message runs. | `ToolCallResult(block=, reason=, arguments=)`. | Argument rewrites chain; the first `block=True` stops the chain. |
| `tool_result` | Once per call, after it runs or is blocked, before the result enters history. | `ToolResultResult(content=, is_error=)`. | Each non-`None` field replaces the current value. |
| `before_compact` | In [compaction](compaction.md), after a cut point is chosen. | `CompactResult(cancel=, cut_index=)`. A `cut_index` is moved forward to the next user message; compaction is skipped if none remains. | Stops at the first `cancel=True`. |
| `turn_end` | The turn has finished. Carries the turn's new messages. | Nothing. | Observation only. |

An event type with no rule in `Hooks._REDUCERS` is observation only. `on` accepts any string, so an extension can use custom event types with its own `emit` calls.

### Tool calls

For an assistant message with several tool calls, the agent emits `tool_call` for every call (concurrently across calls; handlers for one call run in sequence) and applies any rewritten arguments. Only then does it run the calls that were not blocked, using the ordering described in [tools](tools.md#read_only). A blocked call does not run; the model receives an error result carrying the handler's `reason`, or `Blocked by hook` when none is given. Results keep the order of the calls.

A [sub-agent](subagents.md) inherits its parent's `tool_call` and `tool_result` handlers, so a policy applies to delegated calls too. It does not inherit the other events, which were written for the parent's conversation. Observers see only those two events from a child.

### Limits of a hook

A hook sees a tool call's name and arguments before it runs and cannot see or limit what the call then does. A policy over `bash` command strings is advisory, because the same effect can be written many ways. [`examples/approval_extension/`](../examples/approval_extension/) demonstrates a `bash` denylist and is not a safety boundary. For a restriction that holds, allow only tools that cannot do the thing, as [`examples/allowlist_extension/`](../examples/allowlist_extension/) does by leaving `bash` out, and run midge in a container or as a user without the permission.

## The TUI approval prompt

With `[tui] approve_tools` on, the TUI registers its own `tool_call` handler that asks the person before any tool that is not `read_only` runs. It is registered after the extensions loaded at startup, so an extension's block applies before anyone is asked. It has no source, so a profile cannot deactivate it and `reload` does not remove it. RPC never registers it.

## Source-scoped activation

Handlers registered through `register_hooks` are stamped with the extension's path (for diagnostics) and its file stem, the source name. `Hooks.set_active_sources(names)` restricts which sources' handlers run; `None` restores all. Inactive handlers stay registered and are skipped at emit time, so switching a source back on needs no reload. Handlers with no source, such as the TUI approval prompt, always run.

A [profile](profiles.md) sets the active sources through its `hooks` mapping, keyed by stem:

```python
Profile(..., hooks={"approve": True, "allowlist": False})
```

A profile must give a decision for every discovered source or it fails validation, so disabling a guard has to be written explicitly. `Hooks.source_names()` lists the discovered sources.

## Cleanup

Functions registered with `add_cleanup` run when `reload` unloads extensions, when the process exits (after `session_end`), and when `Hooks.clear()` is called. After running the cleanups, `reload` removes every registration that came from an extension and keeps those the entrypoint made itself. See [extensions](extensions.md#reload).
