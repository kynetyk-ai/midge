# py-mono

A hackable Python agent harness originally ported from [`pi-mono`](https://github.com/badlogic/pi-mono) (TypeScript) for personal preference, readability, and domain-adaptability.

The whole harness is roughly 2k LOC. The agent loop, OpenAI-compatible client, tool registry, extension loader, RPC server, HTML exporter, JSONL session save/load, context compaction, and Textual TUI are all small enough to read in a sitting and modify with confidence.

## What's in the box

- **Streaming agent loop** against any OpenAI-compatible endpoint (OpenAI, Azure, ollama, vLLM, LM Studio, llama.cpp's `server`, Together, Groq, Fireworks, OpenRouter, ...).
- **`@tool` decorator** that turns an async Python function into an LLM-callable tool, with JSON Schema generated from its signature via Pydantic.
- **Filesystem extension loader** — drop a `.py` file with `@tool`-decorated functions and an optional `SYSTEM_PROMPT` constant into a directory, point `--extension-dir` at it, and the agent picks up the new tools.
- **Built-in coding tools**: `read`, `write`, `edit`, `bash`.
- **Textual TUI** for interactive use, plus a JSON-on-stdio RPC mode for embedding the agent in external tools.
- **JSONL session save/resume** and a single-file HTML transcript exporter.
- **Context compaction** that summarizes old turns when a session gets long.
- **Retargetable** — `examples/notes_agent.py` demonstrates a non-coding domain (personal-knowledge KB) running on the same harness with no core changes.

## Setup

```bash
poetry install
poetry run pytest
poetry run ruff check
poetry run pyright
```

Python 3.11+. Poetry for env and dep management.

## Use it

### Interactive TUI (coding domain)

```bash
poetry run pym
```

Flags: `--extension-dir DIR` (repeatable), `--session PATH`, `--compaction-threshold N`, `--compaction-keep-recent N`. Bindings: `Ctrl+J` submit, `Ctrl+C` interrupt, `Ctrl+D` quit, `Esc` clear input.

### One-shot CLI

```bash
OPENAI_API_KEY=sk-... \
poetry run python -m examples.coding_agent "list files in cwd"
```

Against a local OpenAI-compatible server:

```bash
OPENAI_BASE_URL=http://127.0.0.1:1234/v1 \
PYM_MODEL=ibm/granite-3.2-8b \
poetry run python -m examples.coding_agent --session run.jsonl --export-html run.html "summarize the README"
```

`PYM_MODEL` defaults to `gpt-4o-mini`. Other flags: `--extension-dir DIR`, `--session PATH` (resumes if file exists), `--export-html PATH`, `--compaction-threshold N`.

### RPC (JSON-on-stdio)

```bash
echo '{"id":"1","type":"prompt","message":"say hi"}' | \
poetry run python -m examples.rpc_agent
```

Newline-delimited JSON; commands `prompt`, `abort`, `get_messages`. Protocol details in [`notes/rpc.md`](./notes/rpc.md).

### Second domain (notes / personal knowledge)

```bash
poetry run python -m examples.notes_agent
```

Same TUI, same agent, no coding tools — just `add_note`, `search_notes`, `read_note`, `list_notes`, `link_notes`. KB lives at `~/.pym-notes/kb.json` by default (override with `PYM_NOTES_KB`).

## Adapting to a new domain

1. Write `.py` files with `@tool`-decorated async functions.
2. Optionally add a module-level `SYSTEM_PROMPT` string to extend the agent's prompt.
3. Drop the directory into `--extension-dir`, or copy `examples/notes_agent.py` and swap in your extension path + system prompt.

The harness deliberately separates from the "coding agent" identity. See `examples/notes_extension/` for a working example, and [`notes/extensions.md`](./notes/extensions.md) for design rationale.

## Layout

```
src/pym/
├── messages.py        # typed message history + OpenAI conversion boundary
├── client.py          # async OpenAI-compatible client + stream events
├── tools/
│   ├── __init__.py    # @tool decorator + Pydantic schema + ToolRegistry
│   └── coding/        # built-in tools: read, bash, edit, write
├── agent.py           # the loop: stream → dispatch tools → repeat
├── compaction.py      # post-turn summarize-and-replace
├── persistence.py     # JSONL session save / load / resume
├── session.py         # single-file HTML transcript exporter
├── rpc.py             # JSON-on-stdio RPC server
├── extensions.py      # load_extensions(dirs) → (ToolRegistry, prompt_addition)
├── tui/app.py         # Textual TUI
└── cli.py             # `pym` entrypoint
examples/
├── coding_agent.py    # one-shot CLI for the coding domain
├── rpc_agent.py       # RPC server for external clients
├── notes_agent.py     # second-domain TUI demo
└── notes_extension/   # the notes extension pack
notes/                 # design rationale + patterns extracted from pi-mono
tests/                 # pytest tests
```

## Reading guide

If you want to understand how the harness works, the files in dependency order:

1. `src/pym/messages.py` — the data model.
2. `src/pym/client.py` — chunk → event mapping; the streaming part.
3. `src/pym/tools/__init__.py` — `@tool` and the registry.
4. `src/pym/agent.py` — the loop.
5. `src/pym/tools/coding/` — four real tools.
6. `src/pym/extensions.py` — the loader.
7. Anything else, in any order: `compaction.py`, `persistence.py`, `session.py`, `rpc.py`, `tui/app.py`.

## License

MIT — see [`LICENSE`](./LICENSE). Copyright (c) 2026 Kynetyk Holdings LLC.

Lineage and dependency credits in [`ACKNOWLEDGEMENTS.md`](./ACKNOWLEDGEMENTS.md).
