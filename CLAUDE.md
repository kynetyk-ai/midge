# py-mono — instructions for Claude

This repo is a Python agent harness originally ported from [`pi-mono`](../pi-mono) (TypeScript). The harness is feature-complete for its original goals (see `README.md`); future work is incremental — bug fixes, polish, new skill packs, individual feature additions.

The user does not work in TypeScript and wants a codebase they can read, modify, and adapt to non-coding domains.

## Goals

1. **Readable, hackable codebase.** Idiomatic Python, not a faithful translation of the TS source.
2. **Learning vehicle.** The user is using this project to understand how an agent harness works internals-up.
3. **Domain-adaptability.** The harness must cleanly separate from the "coding agent" identity. Skills + system prompt should be the only things that need to change to retarget it.

## Working with `pi-mono`

- `../pi-mono/` is **read-only reference material**. Do not edit it.
- `notes/` holds the patterns extracted from `pi-mono` during the original port. When extending a subsystem, check there first to avoid re-reading the same TS code.
- Read for *concepts*, then write Python from scratch. Do not translate line-by-line.

## Tooling and conventions

- **Poetry** for env and dependency management (`poetry install`, `poetry run <cmd>`, `poetry add <pkg>`). Never `pip`, `uv`, `pip-tools`, or `hatch`. `poetry.lock` is committed.
- **Python 3.11+**. Use `asyncio.TaskGroup`, exception groups, and modern type hints.
- **`openai` SDK with configurable `base_url`** is the LLM client. This covers OpenAI plus all OpenAI-compatible local servers (ollama, vLLM, LM Studio, llama.cpp). Do not introduce LangChain or LiteLLM without checking with the user first — the harness loop is small enough that they add weight without buying anything.
- **Pydantic v2** for tool-arg schemas.
- **Textual** for the TUI.
- **Lint:** `ruff`. **Type-check:** `pyright`. **Test:** `pytest` + `pytest-asyncio`.

## Out of scope (do not propose without checking)

- Multi-provider zoo beyond `openai`+`base_url` (Anthropic native, Bedrock, Vertex, Mistral, Azure)
- OAuth flows
- Faithful port of `pi-tui`
- `pi-mom` (Slack), `pi-pods`, `pi-web-ui`
- WASM / native deps
- LangChain / LiteLLM (see above)

## Layout

```
src/pym/            # the harness package
src/pym/skills/     # built-in skills (coding tools live here)
tests/             # pytest tests
examples/          # entrypoints
notes/             # patterns borrowed from pi-mono during reading passes
```

## Style

- No backwards-compatibility scaffolding for code that does not exist yet.
- No comments explaining what code does — only why, when non-obvious.
- Default to no docstrings unless the function is part of a public API.
- Keep abstractions cheap; three similar lines beat a premature helper.
