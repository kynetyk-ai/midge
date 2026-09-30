# Extensions

An extension is a Python file that midge imports at startup to add tools, hooks, sub-agents, profiles and a system-prompt contribution. The loader is [`src/midge/extensions.py`](../src/midge/extensions.py); the built-in coding tools load through it too.

## Writing one

1. Write a `.py` file that defines `@tool` async functions at module top level (see [tools](tools.md)).
2. Optionally add a module-level `SYSTEM_PROMPT` string, a `register_hooks(hooks)` function, `@subagent` declarations or `Profile` instances.
3. Point midge at the file or its directory with `--extension-dir`.

```python
from midge.profiles import Profile
from midge.tools import tool

SYSTEM_PROMPT = "Use the notes tools to capture and find the user's notes."

@tool(read_only=True)
async def search_notes(query: str, limit: int = 10) -> str:
    """Substring search across note titles and bodies."""
    ...

NOTES = Profile(
    name="notes",
    description="A personal-notes assistant.",
    prompt="You are a notes assistant.",
    tools=("search_notes",),
)
```

```bash
midge --extension-dir examples/notes_extension --profile notes
```

[`examples/notes_extension/notes.py`](../examples/notes_extension/notes.py) is a complete domain in one file. [retargeting](retargeting.md) walks through building one.

## What the loader collects

For each source, the loader imports every `.py` file in the directory in name order (not recursively), or the single file named. Files starting with `_` and `__init__.py` are skipped. From each module's top-level names it collects:

| Declaration | Effect |
|---|---|
| `Tool` instances (from `@tool`, or a `spawn_*` tool from `@subagent`) | Added to the agent's tool registry. See [tools](tools.md) and [subagents](subagents.md). |
| `SYSTEM_PROMPT` (non-empty `str`) | Appended to the system prompt. |
| `register_hooks(hooks)` | Called once with the hook registry. See [hooks](hooks.md). |
| `Profile` instances | Registered as selectable profiles. See [profiles](profiles.md). |

Detection is by type: the loader looks for `Tool` and `Profile` objects in the module namespace, so no registration call is needed.

Sources load in this order: the built-in tools (as selected by `[tools] builtin`), the configured extension directory if enabled, then each `--extension-dir` in command-line order. When two files declare a tool or a profile with the same name, the first wins and the later one is dropped with a `tool_name_shadowed` or `profile_name_shadowed` warning. An extension therefore cannot replace a built-in by reusing its name.

A file that fails to import, or whose `register_hooks` raises, is logged with its traceback and skipped. Startup continues.

Unless a file defines its own module-level `log`, the loader sets one to the logger `midge.ext.<file stem>`, so an extension's records go through midge's logging configuration (see [logging](logging.md)). A logger the extension creates itself is covered by that configuration only if its name is under `midge`. Use it instead of `print()`: stdout is the protocol in RPC mode, and the TUI discards it.

## The system prompt

Each module's `SYSTEM_PROMPT` is stripped and joined with blank lines in load order. The combined text follows the base or profile prompt and precedes the skills catalogue. It describes the tools an extension adds; the agent's identity belongs to a profile or `[agent] system_prompt`. The RPC command `get_system_prompt` returns the composed prompt and its parts (see [rpc](rpc.md)).

## Loading from config

`--extension-dir` is honoured on every run. To load a directory without the flag, set `[extensions] enabled = true` in the config file. The directory is `[extensions] dir`, defaulting to `.agents/extensions` under the working directory.

Autoload is off by default because an extension is arbitrary Python executed in the midge process at startup. It can register tools, add hooks that block or rewrite tool calls, and declare profiles that change which hooks are active. A project's `.midge/config.toml` can set this table, so enabling it in a repository you did not write runs that repository's code. See [`examples/config.toml`](../examples/config.toml) for the keys and their environment variables.

## Reload

The `reload` command (over RPC, or `/reload` in the TUI) re-reads extensions and skills from disk. It is refused while a turn is running. For extensions it:

1. runs every cleanup registered with `hooks.add_cleanup`, then removes every hook registered through an extension; handlers registered by the entrypoint itself, such as the TUI's approval prompt, stay;
2. re-imports every source, producing a fresh tool registry, fresh profiles and fresh hook registrations;
3. re-validates profiles, re-binds sub-agents, and reapplies the active profile so the agent is not widened to every tool on disk;
4. recomposes the system prompt.

A re-import creates a new module object. The previous module's globals are not reclaimed, and anything it opened stays open unless it registered a cleanup. An extension that holds a resource (a file handle, a connection, a background task) should close it from `hooks.add_cleanup`.
