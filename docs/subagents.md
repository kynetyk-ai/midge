# Sub-agents

A sub-agent is a nested agent the model delegates work to by calling a `spawn_<name>` tool. The
child runs with its own system prompt and tool allowlist, and only its final answer returns to the
parent's context. The code is [`src/midge/subagents.py`](../src/midge/subagents.py).

Delegation happens inside one midge process. Supervising several midge processes from outside is a
separate concern that midge does not provide. A sub-agent is also distinct from a
[profile](profiles.md): a sub-agent is a tool the agent calls, while a profile reconfigures the
agent itself.

## Declaring one

Decorate an `async` function with `@subagent` in an extension `.py` file:

```python
from midge.subagents import subagent

@subagent(
    description="Locate where something lives in the codebase. Read-only.",
    prompt="You are a code explorer. Cite path:line for every claim.",
    tools=("read", "ls", "grep"),
    timeout=180,
)
async def explore(question: str, paths: list[str] | None = None) -> str:
    scope = "\n".join(paths or ["(whole repository)"])
    return f"Question: {question}\n\nStart from:\n{scope}"
```

```bash
midge --extension-dir examples/subagent_extension
```

`@subagent` returns a `Tool`, so the ordinary [extension](extensions.md) loader discovers it with
`--extension-dir` and there is no separate agent directory. The tool is named `spawn_` plus the
function name, or plus `name=` if given. The decorator's arguments:

| Argument | Meaning |
|---|---|
| `description` | The tool description the model sees when deciding whether to delegate |
| `prompt` | The child's system prompt |
| `tools` | Names of the parent's tools the child may call. Empty means none |
| `model` | The child's model. Empty means the parent's current model |
| `timeout` | Seconds the child may run. Unset means `[subagents] timeout` |
| `name` | Overrides the function name as the agent's name |

A complete example is [`examples/subagent_extension/`](../examples/subagent_extension/).

## The signature is the schema

The function's parameters become the `spawn_*` tool's argument schema, built the same way as for
`@tool` (see [tools](tools.md)): type hints, defaults, and `Annotated[..., Field(description=...)]`
all apply, and unknown arguments are refused. When the model calls the tool, midge validates the
arguments, calls the function, and sends its return value to the child as the opening user message.

The model therefore supplies only the declared inputs. The child's system prompt, tools and model
come from the declaration and cannot be set by the caller.

## What the child can do

The child's tool registry is the subset of the parent's registry named in `tools`. A tool outside
the allowlist is absent from the child's schemas and refused if called.

A child can spawn further sub-agents only if its allowlist names their `spawn_*` tools. There is no
global depth limit. A child never receives a `spawn_*` tool for an agent already running above it,
so an allowlist cycle such as `alpha -> beta -> alpha` is cut where it would close, and nesting
always terminates.

### `read_only` is derived

A `spawn_*` tool's `read_only` is computed from its allowlist: it is true only if every tool the
child may call is read-only, following nested `spawn_*` tools. An explorer allowed `bash` is
therefore treated as a state-changing tool, whatever its description says. The flag decides whether
the call may run alongside other read-only calls and whether the TUI asks before running it (see
[tools](tools.md)).

### Hooks

The child inherits the parent's `tool_call` and `tool_result` [hooks](hooks.md), so an approval or
blocking policy applies to delegated tool calls as well. All other events are withheld from the
child, because handlers written for the parent's conversation would otherwise act on the child's
(for example, a `turn_start` handler that sets `system_prompt` would replace the child's prompt).
The filter applies to `observe()` handlers too, so an audit observer still sees the child's tool
calls. The parent's `tool_call` hook also sees the `spawn_*` call itself and can block it by name.

## Bounds

Every run has a wall-clock budget, chosen in this order:

1. A `timeout` argument from the caller, if the function declares a `timeout` parameter. Declaring
   it is what puts it in the schema.
2. The `timeout` passed to `@subagent`.
3. `[subagents] timeout`.

The result is capped at `[subagents] max_timeout`. On timeout the tool returns
`[<name> timed out after <n>s]`.

`[subagents] max_concurrent` limits how many children run at once across the whole process. Waiting
runs queue on the limit. Nested children count against the same limit while their parent holds a
slot, so a limit at or below the nesting depth you allow can leave a grandchild waiting until its
parent times out.

Several `spawn_*` calls in one model response run concurrently when they are read-only, and one at
a time otherwise, the same as any other tool. A child runs inline in the parent's tool call, so
interrupting the parent's turn ends the child with it.

See [`examples/config.toml`](../examples/config.toml) for the `[subagents]` keys and
[config](config.md) for how they are resolved.

## Results

The parent receives the text of the child's final assistant message. If the child ends with an
error or is aborted, the result is `[<name> did not finish: <reason>]`; if it produces no text, the
result is `[<name> returned nothing]`.

## Transcripts

When the parent has a [session](sessions.md), each run writes its own transcript beside the
parent's, named `<parent-stem>.<agent>-<call-id>.jsonl` after the tool call that spawned it. An id
that is empty or already taken falls back to a numbered name. The link is recorded in both
directions:

- the child's header has `origin: "subagent"`, `parent_session` and `parent_tool_call_id`;
- the parent's transcript gets a `continued` record naming the child file.

The transcript is opened only once the run acquires a concurrency slot, and is written even when
the run times out or is interrupted. `list_sessions` omits these files by default, since reopening
one would resume the middle of a tool call. With no parent session (`--no-session`), no child
transcript is written.

## Live events

In [RPC](rpc.md) mode, a child's tool executions, errors and end are forwarded to the client with
an `agent` envelope (`agent`, `agent_id`, `parent_id`, `depth`). `agent_id` is the spawning tool
call's id, the same value the child's transcript records as `parent_tool_call_id`. The child's
streamed text is not forwarded. [`docs/rpc.md`](rpc.md#from-a-sub-agent) has the frame format.

## Validation

After all extensions load, midge checks the declared sub-agents and reports problems as startup
diagnostics:

- a `tools` entry naming no registered tool is a warning;
- an allowlist cycle is a warning;
- a `model` that no `[models]` entry names drops the agent, because it could not run. An empty
  `[models]` registry accepts any model (see [providers](providers.md)).

## Binding

A tool cannot see the agent that called it, so each entrypoint calls
`bind_subagents(registry, client=..., model=..., hooks=..., session=..., subagents=...)` to give
every `spawn_*` tool what it needs to run a child. It does nothing on a registry without sub-agents.
`Controls.bind_subagents` in [`commands.py`](../src/midge/commands.py) re-binds after `reload`,
`new_session`, `open_session` and a profile switch, so a custom entrypoint that uses `Controls` needs no
extra code for this. [`examples/coding_agent.py`](../examples/coding_agent.py) shows a direct call.
