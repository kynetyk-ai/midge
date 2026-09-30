# Tools

A tool is an async Python function the model can call. The `@tool` decorator turns the function into a `Tool` with a JSON schema; a `ToolRegistry` holds the tools an agent offers. Both live in [`src/midge/tools/__init__.py`](../src/midge/tools/__init__.py).

## Declaring a tool

```python
from midge.tools import tool

@tool(read_only=True)
async def word_count(path: str, unique: bool = False) -> str:
    """Count the words in a text file."""
    ...
```

- The function must be `async def`; the decorator raises `TypeError` otherwise.
- The tool's name is the function name and its description is the docstring. `name=` and `description=` override either.
- The parameters come from the signature. Each annotation becomes a field of a generated Pydantic model, a parameter with a default is optional, and unknown arguments are rejected. A Pydantic model is a valid annotation for structured arguments (`edit` takes a `list[EditOp]`). `*args` and `**kwargs` are not supported.
- `Tool.schema()` returns `{"name", "description", "parameters"}`, with `parameters` taken from the model's JSON schema.

A tool reports failure by raising. The agent turns the exception into an error result for the model and logs it with its traceback. A name that is not registered (`ToolNotFound`) and arguments that fail validation (`InvalidArguments`) are reported separately from an exception raised inside the tool body, so the model can tell a wrong call from a failing tool. The agent converts the return value with `str()` and truncates results longer than `MAX_TOOL_RESULT_CHARS` in [`agent.py`](../src/midge/agent.py); the built-in tools bound their own output below that.

Tools reach midge through an extension file (see [extensions](extensions.md)). A tool cannot see the agent that called it. A `Tool` subclass that needs to tie its output to the call receives the provider's `call_id` in `invoke`.

## `read_only`

`@tool(read_only=True)` declares that the tool only observes. It controls two things:

- **Execution order.** When one assistant message contains several tool calls, the agent runs consecutive read-only calls concurrently. Any other call runs alone, after every call before it has finished and before any call after it starts (`Agent._run_in_order`). A model that writes a file and then edits it in the same message gets the edit it asked for.
- **Approval.** With `[tui] approve_tools` on, the TUI asks the person before running any tool that is not read-only. RPC never asks.

The default is `False`, because an unclassified tool that is serialized and confirmed costs time, while a mutating tool marked read-only can race another call and runs without approval. Declare `True` only when the tool has no side effects. A `spawn_*` tool derives the value from its allowlist: it is read-only only when every tool the sub-agent may call is (see [subagents](subagents.md)).

`read_only` concerns scheduling and prompting. It does not restrict what a tool can do; see [hooks](hooks.md) for gating calls.

## `reads_paths`

`@tool(reads_paths=True)` declares that the tool takes an absolute file path and returns the file's text. The skills catalogue is shown to the model only when some registered tool declares it (see [skills](skills.md)). The built-in `read` does; a domain without the coding tools can provide its own reader.

## `ToolRegistry`

A registry maps names to tools. `add` rejects a duplicate name, `remove` is silent on a missing one, `schemas()` returns what the provider sends to the model, and `invoke(name, arguments)` validates and runs a call. The agent's registry is `Agent.tools`. A profile narrows it to a subset (see [profiles](profiles.md)), and `reload` replaces it (see [extensions](extensions.md#reload)).

## Built-in tools

The coding tools live in [`src/midge/tools/coding/`](../src/midge/tools/coding/) and load through the same extension loader as any other tool. Relative paths resolve against the process's working directory. The tools do not confine paths; [`examples/allowlist_extension/`](../examples/allowlist_extension/) shows how to restrict writes with a hook.

| Tool | Read-only | Behaviour |
|---|---|---|
| `read` | yes | Reads a UTF-8 file from a 1-indexed `offset`. Output is head-truncated by line and byte limits, with a note giving the offset to continue from. Declares `reads_paths`. |
| `ls` | yes | Lists a directory alphabetically, directories marked with `/`, dotfiles included, up to `limit` entries. |
| `grep` | yes | Searches file contents with a Python regex (or `literal=true`), optionally filtered by `glob`. Inside a git repository it searches only tracked and unignored files. Skips binary files and stops at `limit` matches or a byte cap. |
| `write` | no | Writes UTF-8 content, overwriting, and creates parent directories. |
| `edit` | no | Applies a list of `{old_text, new_text}` replacements. Each `old_text` must match exactly once in the original content, and edits may not overlap. Preserves a BOM and CRLF line endings. Returns a unified diff and the first changed line. When a match fails, the error describes the nearest text in the file but applies nothing. |
| `bash` | no | Runs `/bin/bash -c` with stdout and stderr interleaved, in its own process group, with a `timeout` in seconds. On timeout or cancellation the group is killed. Output is tail-truncated; over the byte cap, the full output is written to a temp file whose path is included in the result. A non-zero exit code is appended to the output. |

Each tool's docstring states its exact limits and is the description the model sees.

## Choosing built-ins: `[tools] builtin`

`[tools] builtin` in the config file selects which built-ins load: `true` (all), `false` (none, for a domain that supplies its own tools), or a list of names. A name that is not a built-in is logged as a warning and ignored. Extension tools are unaffected by this setting. Keep `read`, or another `reads_paths` tool, if the domain uses skills. See [`examples/config.toml`](../examples/config.toml) for the entry and its environment variable, and [config](config.md) for precedence.

Built-ins load before any extension, and on a name collision the first registration wins, so an extension cannot replace a built-in by reusing its name. To substitute your own implementation, drop the built-in with `[tools] builtin` and register the replacement from an extension.
