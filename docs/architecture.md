# Architecture

How midge's modules fit together, and an order for reading the source.

## The pieces

```
cli.py ──> Config ──> extensions, skills, profiles ──> Agent ──> Controls ──> front-end
                                                         │                     (rpc/ or tui/)
                                                         └──> Client ──> provider adapter
```

- **Entrypoint** (`src/midge/cli.py`). Loads [configuration](config.md), configures [logging](logging.md), loads [extensions](extensions.md) and [skills](skills.md), resolves the [profile](profiles.md) and [session](sessions.md), builds the `Agent`, and hands it to a front-end. Only entrypoints read configuration or configure logging; library modules take parameters.
- **Messages** (`src/midge/messages.py`). The provider-independent history: user, assistant and tool-result messages. Everything else reads and writes these types.
- **Client and providers** (`src/midge/client.py`, `src/midge/providers/`). The client runs one streaming request with retries and yields normalized events. A provider adapter owns a vendor's wire format and error semantics. See [providers](providers.md).
- **Tools** (`src/midge/tools/`). `@tool` turns an async function into a tool with a JSON Schema. The built-in coding tools live in `tools/coding/`. See [tools](tools.md).
- **Agent loop** (`src/midge/agent.py`). Streams a response, runs the tool calls in it, appends results, and repeats until the model stops calling tools. [Hooks](hooks.md) can intercept each step. See [agent loop](agent-loop.md).
- **Extensions** (`src/midge/extensions.py`, `subagents.py`, `profiles.py`, `hooks.py`). User `.py` files that add tools, hooks, [sub-agents](subagents.md) and profiles.
- **Session and compaction** (`src/midge/persistence.py`, `compaction.py`). The JSONL transcript and the summarization that keeps a long session within the context window. See [sessions](sessions.md) and [compaction](compaction.md).
- **Controls** (`src/midge/commands.py`). The operations a person or client can perform on a running agent (compact, switch model or profile, open a session, reload) and the command table both front-ends list. A front-end translates its input into a `Controls` call and renders the result.
- **Front-ends** (`src/midge/rpc/`, `src/midge/tui/`). [RPC](rpc.md) speaks newline-delimited JSON on stdio. The TUI is a Textual app. Anything that needs a person, such as tool approval, lives in `tui/` so it can never block RPC.

## Reading order

To learn how the harness works from the inside, read the source in dependency order:

1. `src/midge/messages.py`, the data model.
2. `src/midge/client.py`, then `src/midge/providers/`.
3. `src/midge/tools/__init__.py`, the decorator and registry.
4. `src/midge/agent.py`, the loop.
5. `src/midge/tools/coding/`, the built-in tools.
6. `src/midge/extensions.py`, the loader.
7. `src/midge/hooks.py`, lifecycle interception.
8. The rest in any order: `compaction.py`, `persistence.py`, `commands.py`, `rpc/`, `tui/`.
