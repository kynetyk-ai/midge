# midge

A hackable Python agent harness, originally ported from [`pi-mono`](https://github.com/badlogic/pi-mono) (TypeScript), built for readability and for adapting to domains other than coding.

The agent loop, provider client, tool registry, extension loader, RPC server, session files, compaction and Textual TUI are each small enough to read in a sitting. CI holds the core harness (`src/midge/*.py`) under 5,000 lines of code; `poetry run python scripts/loc.py` prints the per-file table.

## What's in the box

- **Streaming agent loop** against OpenAI's Responses API or any chat-completions server (ollama, vLLM, LM Studio, llama.cpp, OpenRouter and others), with retries that respect `Retry-After` and a model registry for using several services at once. See [agent loop](docs/agent-loop.md) and [providers](docs/providers.md).
- **`@tool`** turns an async Python function into a tool, with its schema generated from the signature. Built-in coding tools: `read`, `ls`, `grep`, `write`, `edit`, `bash`. See [tools](docs/tools.md).
- **Extensions**: `.py` files that add tools, hooks, sub-agents and profiles, loaded with `--extension-dir`. See [extensions](docs/extensions.md).
- **[Agent Skills](https://agentskills.io/specification)**: `SKILL.md` directories of instructions, listed in the system prompt and read when a task matches. See [skills](docs/skills.md).
- **Sub-agents**: a nested agent the model delegates to through a `spawn_<name>` tool, with its own prompt and a subset of the parent's tools. See [sub-agents](docs/subagents.md).
- **Profiles**: named configurations (prompt, model, tools, active hooks) the agent runs as. See [profiles](docs/profiles.md).
- **Hooks** that can block or rewrite tool calls, patch results and reshape requests. See [hooks](docs/hooks.md).
- **Sessions** recorded to an append-only JSONL transcript by default, and **compaction** that summarizes old turns in long sessions. See [sessions](docs/sessions.md) and [compaction](docs/compaction.md).
- **Two front-ends**: a Textual TUI, and a JSON-on-stdio [RPC protocol](docs/rpc.md) for driving midge from other programs.
- **Retargetable**: a non-coding domain is an extension, a profile and a little config, run through the same entrypoint. See [retargeting](docs/retargeting.md).

## Quick start

```bash
pipx install "midge[tui] @ git+https://github.com/kynetyk-ai/midge@v0.1.0"
export OPENAI_API_KEY=sk-...
```

Without `[tui]` only `midge --rpc` is available.

```bash
cd ~/code/my-project
midge                                        # the TUI
midge --rpc                                  # the RPC server on stdin/stdout
midge --extension-dir ~/coding/midge/examples/notes_extension --profile notes   # a notes assistant
```

[Running midge](docs/running.md) covers the TUI's keys and commands, containers, Docker Sandboxes and the one-shot CLI. [Configuration](docs/config.md) covers `.midge/config.toml`, models and providers.

## Documentation

[`docs/`](docs/README.md) has a doc per subsystem, the RPC protocol, a retargeting guide, and an [architecture overview](docs/architecture.md) with an order for reading the source.

## Development

```bash
poetry install
poetry run pytest
poetry run ruff check
poetry run pyright
```

Python 3.11+, Poetry for environment and dependency management. Behaviour changes are tested against a real model by the end-to-end tests before merge; see [`e2e/README.md`](e2e/README.md).

## Layout

```
src/midge/
├── messages.py        # typed message history (provider-independent)
├── client.py          # streaming state machine + retry policy
├── providers/         # one adapter per wire format; the model registry; Delta contract
├── tools/
│   ├── __init__.py    # @tool decorator + Pydantic schema + ToolRegistry
│   └── coding/        # built-in tools: read, ls, grep, write, edit, bash
├── agent.py           # the loop: stream → dispatch tools → repeat
├── compaction.py      # post-turn summarize-and-replace
├── persistence.py     # JSONL session save / load / resume
├── commands.py        # Controls: the operations both front-ends call
├── rpc/               # JSON-on-stdio front-end: wire / server / transport
├── extensions.py      # load_extensions(dirs) → (ToolRegistry, prompt_addition)
├── config.py          # .midge/config.toml → a Config the entrypoint passes inward
├── logs.py            # logging config; entrypoints only
├── skills.py          # SKILL.md discovery + <available_skills> catalogue
├── subagents.py       # @subagent → spawn_* tools running nested agents
├── profiles.py        # Profile discovery + validation (what the agent *is*)
├── hooks.py           # lifecycle events + handler registry
├── tui/app.py         # Textual TUI: commands (Ctrl+P), drawer (Ctrl+B), slash commands, steering
└── cli.py             # `midge` entrypoint
examples/
├── coding_agent.py    # one-shot CLI for the coding domain
├── rpc_client.py      # a minimal client for `midge --rpc` (stdlib only)
├── config.toml        # every config key, commented, with its default
├── approval_extension/ # tool-approval hook demo (a denylist: advisory)
├── allowlist_extension/ # a restriction that holds: named tools only, writes under cwd
├── notes_extension/   # the notes extension pack
├── subagent_extension/ # a read-only explorer sub-agent
├── profile_extension/ # a declared profile: the adversarial reviewer
└── skills/            # a worked SKILL.md example
sandbox/midge/         # Docker Sandboxes kit: image, kit spec, sandbox-env skill
docs/                  # subsystem docs, guides, ADRs (index: docs/README.md)
tests/                 # pytest tests
```

## License

MIT; see [`LICENSE`](./LICENSE). Copyright (c) 2026 Kynetyk Holdings LLC.

Lineage and dependency credits in [`ACKNOWLEDGEMENTS.md`](./ACKNOWLEDGEMENTS.md).
