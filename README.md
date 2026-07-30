# midge

A hackable Python agent harness originally ported from [`pi-mono`](https://github.com/badlogic/pi-mono) (TypeScript) for personal preference, readability, and domain-adaptability.

The core harness is under 3.5k lines of code — blank lines, comments and docstrings excluded — and **CI holds it under 5k** (`scripts/loc.py`). The agent loop, OpenAI-compatible client, tool registry, extension loader, RPC server, HTML exporter, JSONL session save/load, context compaction, and Textual TUI are all small enough to read in a sitting and modify with confidence.

The budget covers `src/midge/*.py`, the harness itself. `providers/`, `tools/` and `tui/` are excluded and reported separately: a provider adapter grows with the number of vendors, built-in tools with the domain, and the TUI is a presentation layer. Prose is free, so hitting the cap means simplifying code rather than deleting the explanation of why it works that way. Run `poetry run python scripts/loc.py` for the per-file table.

## What's in the box

- **Streaming agent loop** against any OpenAI-compatible endpoint (OpenAI, Azure, ollama, vLLM, LM Studio, llama.cpp's `server`, Together, Groq, Fireworks, OpenRouter, ...). The wire format lives behind a provider registry, so the streaming state machine and retry policy are written once and a second format is an adapter rather than a branch.
- **`@tool` decorator** that turns an async Python function into an LLM-callable tool, with JSON Schema generated from its signature via Pydantic.
- **Filesystem extension loader** — drop a `.py` file with `@tool`-decorated functions and an optional `SYSTEM_PROMPT` constant into a directory, point `--extension-dir` at it, and the agent picks up the new tools.
- **Agent Skills** ([`SKILL.md`](https://agentskills.io/specification)) — drop a directory of markdown instructions in and point `--skill-dir` at it. Names and descriptions go in the system prompt; the agent opens the full file with `read` only when a task matches. No Python, no prompt edits, and directories written for other harnesses load as-is.
- **Sub-agents** — declare a nested agent in a `.py` file and it becomes a `spawn_<name>` tool the model can delegate to, with its own system prompt and a subset of the parent's tools. The parent gets the result; the child's own turns stay out of its context and go to a linked transcript.
- **Built-in coding tools**: `read`, `write`, `edit`, `bash`.
- **Lifecycle hooks** — block or rewrite a tool call before it runs, transform context, patch results, observe every event. See [`notes/hooks.md`](./notes/hooks.md) and `examples/approval_extension/`.
- **Textual TUI** for interactive use, plus a JSON-on-stdio RPC mode for embedding the agent in external tools.
- **JSONL session save/resume** and a single-file HTML transcript exporter. The format is append-only: a rename or a context clear is a record appended and replayed on load, never a rewrite, so a crash can only ever damage the final line.
- **Context compaction** that summarizes old turns when a session gets long.
- **Provider retry** with a cancellable backoff — rate limits, 5xx, and transport failures get a few attempts before the turn fails. Retries stop once the response has started streaming, so nothing the model already emitted is replayed.
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
poetry run midge
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
MIDGE_MODEL=ibm/granite-3.2-8b \
poetry run python -m examples.coding_agent --session run.jsonl --export-html run.html "summarize the README"
```

`model` defaults to `gpt-4o-mini` — see [Configuration](#configuration). Other flags: `--extension-dir DIR`, `--skill-dir DIR`, `--skill NAME`, `--session PATH` (resumes if file exists), `--export-html PATH`, `--compaction-threshold N`.

### Configuration

Everything configurable lives in one TOML file, read from `./.midge/config.toml` then
`~/.midge/config.toml` and merged key by key — so a project file that pins a model does not
discard your personal log level. [`examples/config.toml`](./examples/config.toml) lists every
key with its default.

```toml
model = "ibm/granite-3.2-8b"

[provider]
base_url = "http://127.0.0.1:1234/v1"

[log]
level = "INFO"
file = "~/.midge/midge.log"
```

Precedence is **flag > environment variable > config file > default**, so `MIDGE_LOG_LEVEL=DEBUG`
still wins for one run and CI keeps working unchanged. Every key has an environment variable
equivalent; an unrecognized key is reported at startup rather than ignored, and a malformed file
is a warning, not a failure to start.

### The model registry

Optional, and empty by default. Listing models lets midge talk to **more than one service at
once** — the model determines where a request goes, so `set_model` and a sub-agent's `model=`
route correctly instead of all landing on one endpoint:

```toml
[providers.openai]
api_key_env = "OPENAI_API_KEY"

[providers.local]
kind = "openai-compatible"
base_url = "http://localhost:11434/v1"

[models."gpt-4o-mini"]
provider = "openai"

[models."ibm/granite-3.2-8b"]
provider = "local"
```

An **empty registry is permissive** — any model string works, which is what every install without
these tables gets. Writing even one `[models]` entry turns enforcement on: from then on you are
declaring what you want available, and `set_model` refuses anything else rather than reporting
success and failing on the next turn.

midge ships **no list of models** and does not check that a model id exists. Vendors add and
retire models continuously, so a baked-in list is wrong within weeks, and a typo fails at the API
with the vendor's own error. What is checked is the wiring: a model naming a provider you never
defined is dropped with a warning at startup.

`OPENAI_API_KEY` is the one setting that is **not** a config key. A credential does not belong in
a file that gets committed, so it is read from the environment, in exactly one place, and never
logged at any level. A `base_url` is logged as its hostname only, because it can carry
credentials in its userinfo.

Only entrypoints construct a `Config`; library modules take parameters. There is no
`get_config()` to import — see [`src/midge/config.py`](./src/midge/config.py) for why.

### RPC (JSON-on-stdio)

```bash
echo '{"id":"1","type":"prompt","message":"say hi"}' | \
poetry run python -m examples.rpc_agent
```

Newline-delimited JSON. `get_commands` enumerates everything invocable — built-in commands and `SKILL.md` skills alike — each with a JSON Schema for its arguments, so a client can build a command palette without hardcoding the protocol. `reload` re-scans skills and extensions from disk, so a new `SKILL.md` or an edited tool takes effect without restarting. Protocol details in [`notes/rpc.md`](./notes/rpc.md).

### Second domain (notes / personal knowledge)

```bash
poetry run python -m examples.notes_agent
```

Same TUI, same agent, no coding tools — just `add_note`, `search_notes`, `read_note`, `list_notes`, `link_notes`. KB lives at `~/.midge-notes/kb.json` by default (override with `MIDGE_NOTES_KB`).

## Adapting to a new domain

There are two levers, and they compose. **Extensions** add capabilities the agent
did not have; **skills** teach it how to use the capabilities it already has.

### Skills — markdown, no Python

A skill is a directory with a `SKILL.md`: YAML frontmatter carrying a `name` and
a `description`, then instructions. Anything else beside it is freeform.

```
my-skills/
└── commit-message/
    ├── SKILL.md
    └── references/
        └── style.md
```

```markdown
---
name: commit-message
description: >-
  Writes a git commit message for the staged changes, following the project's
  conventions. Use when the user asks to commit or to describe a change.
---

# Commit message

Run `git diff --cached` first. Read `references/style.md` for the full rules.
```

```bash
poetry run python -m examples.coding_agent --skill-dir my-skills "commit this"
```

Only the name and description sit in the system prompt. The agent reads the full
file with the `read` tool when a task matches its description, and resolves
`references/style.md` against the skill's own directory — so a long reference
document costs nothing until it is actually needed.

Skills are also discovered from `./.midge/skills`, `./.agents/skills`,
`~/.midge/skills` and `~/.agents/skills` without any flag. `--skill-dir` entries
win a name collision, then project directories, then personal ones.

Validation is lenient so directories written for other harnesses load unchanged —
`--skill-dir ~/.claude/skills` works. A bad name or an over-long description
warns and still loads; only a missing `description` is fatal, since it is the
only thing the model sees before deciding whether to open the file.

Models do not always take the hint. `--skill NAME` forces one: its body is sent
as the turn and the prompt becomes the instructions.

```bash
poetry run python -m examples.coding_agent --skill-dir examples/skills \
  --skill commit-message "keep it to one line"
```

Add `disable-model-invocation: true` to a skill's frontmatter to keep it out of
the catalogue entirely, leaving `--skill` as the only way in.

See `examples/skills/` for a worked example and [`notes/skills.md`](./notes/skills.md)
for design rationale.

### Sub-agents — delegate work out of the conversation

A sub-agent is a tool. Declare one in a `.py` file and the model sees `spawn_<name>` alongside
`read` and `bash`:

```python
from midge.subagents import subagent

@subagent(
    description="Locate where something lives in the codebase. Read-only.",
    prompt="You are a code explorer. Cite path:line for every claim.",
    tools=("read", "bash"),
    timeout=180,
)
async def explore(question: str, paths: list[str] | None = None) -> str:
    scope = "\n".join(paths or ["(whole repository)"])
    return f"Question: {question}\n\nStart from:\n{scope}"
```

```bash
poetry run python -m examples.coding_agent --extension-dir examples/subagent_extension \
  "where does compaction choose its cut point?"
```

The decorated function's **signature is the tool schema** and its **return value is the child's
opening message** — so the model supplies the declared inputs and nothing else. It cannot choose
the child's system prompt, its tools, or its model.

Only the child's final answer reaches the parent, which is the point: a search that reads thirty
files costs the main conversation one paragraph. With `--session`, the child's own turns go to a
sibling transcript named for the tool call that spawned it, so the delegated work stays
inspectable and unambiguously linked.

A sub-agent inherits the parent's **tool policy** — an approval hook that blocks a command blocks it
when delegated too — but not the parent's prompt- or request-shaping hooks, which would otherwise
silently override the child's own system prompt. Nesting is capped by depth, and a child only gets
`spawn_*` tools if its allowlist names them.

See `examples/subagent_extension/` and [`notes/subagents.md`](./notes/subagents.md).

### Extensions — new tools

1. Write `.py` files with `@tool`-decorated async functions.
2. Optionally add a module-level `SYSTEM_PROMPT` string to extend the agent's prompt.
3. Drop the directory into `--extension-dir`, or copy `examples/notes_agent.py` and swap in your extension path + system prompt.

The harness deliberately separates from the "coding agent" identity. See `examples/notes_extension/` for a working example, and [`notes/extensions.md`](./notes/extensions.md) for design rationale.

## Layout

```
src/midge/
├── messages.py        # typed message history (provider-independent)
├── client.py          # streaming state machine + retry policy
├── providers/         # one adapter per wire format; the model registry; Delta contract
├── tools/
│   ├── __init__.py    # @tool decorator + Pydantic schema + ToolRegistry
│   └── coding/        # built-in tools: read, bash, edit, write
├── agent.py           # the loop: stream → dispatch tools → repeat
├── compaction.py      # post-turn summarize-and-replace
├── persistence.py     # JSONL session save / load / resume
├── session.py         # single-file HTML transcript exporter
├── rpc.py             # JSON-on-stdio RPC server
├── extensions.py      # load_extensions(dirs) → (ToolRegistry, prompt_addition)
├── config.py          # .midge/config.toml → a Config the entrypoint passes inward
├── logs.py            # logging config; entrypoints only
├── skills.py          # SKILL.md discovery + <available_skills> catalogue
├── subagents.py       # @subagent → spawn_* tools running nested agents
├── hooks.py           # lifecycle events + handler registry
├── tui/app.py         # Textual TUI
└── cli.py             # `midge` entrypoint
examples/
├── coding_agent.py    # one-shot CLI for the coding domain
├── rpc_agent.py       # RPC server for external clients
├── notes_agent.py     # second-domain TUI demo
├── config.toml        # every config key, commented, with its default
├── approval_extension/ # tool-approval hook demo
├── notes_extension/   # the notes extension pack
├── subagent_extension/ # a read-only explorer sub-agent
└── skills/            # a worked SKILL.md example
notes/                 # design rationale + patterns extracted from pi-mono
tests/                 # pytest tests
```

## Reading guide

If you want to understand how the harness works, the files in dependency order:

1. `src/midge/messages.py` — the data model.
2. `src/midge/client.py` — the streaming state machine, then `providers/` for the wire format it is fed by.
3. `src/midge/tools/__init__.py` — `@tool` and the registry.
4. `src/midge/agent.py` — the loop.
5. `src/midge/tools/coding/` — four real tools.
6. `src/midge/extensions.py` — the loader.
7. `src/midge/hooks.py` — lifecycle interception.
8. Anything else, in any order: `compaction.py`, `persistence.py`, `session.py`, `rpc.py`, `tui/app.py`.

## License

MIT — see [`LICENSE`](./LICENSE). Copyright (c) 2026 Kynetyk Holdings LLC.

Lineage and dependency credits in [`ACKNOWLEDGEMENTS.md`](./ACKNOWLEDGEMENTS.md).
