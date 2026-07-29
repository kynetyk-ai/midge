# midge — instructions for Claude

This repo is a Python agent harness originally ported from [`pi-mono`](../pi) (TypeScript). The harness is feature-complete for its original goals (see `README.md`); future work is incremental — bug fixes, polish, new extension packs, individual feature additions.

The user does not work in TypeScript and wants a codebase they can read, modify, and adapt to non-coding domains.

## Goals

1. **Readable, hackable codebase.** Idiomatic Python, not a faithful translation of the TS source.
2. **Learning vehicle.** The user is using this project to understand how an agent harness works internals-up.
3. **Domain-adaptability.** The harness must cleanly separate from the "coding agent" identity. Extensions + system prompt should be the only things that need to change to retarget it.

## Working with `pi-mono`

- `../pi/` is **read-only reference material** (the `pi-mono` repo, checked out as `pi`). Do not edit it.
- `notes/` holds the patterns extracted from `pi-mono` during the original port. When extending a subsystem, check there first to avoid re-reading the same TS code.
- Read for *concepts*, then write Python from scratch. Do not translate line-by-line.

### Vocabulary — deliberately aligned with `pi-mono`

| Concept | midge | `pi-mono` |
|---|---|---|
| Built-in LLM-callable tools (`read`, `bash`, …) | `src/midge/tools/coding/` | `packages/coding-agent/src/core/tools/` |
| Loading user `.py` files that register tools | `src/midge/extensions.py`, `--extension-dir` | `src/core/extensions/` |
| `SKILL.md` — the [Agent Skills standard](https://agentskills.io/specification) | `src/midge/skills.py`, `--skill-dir` | `src/core/skills.ts` |

The word **skill** means the `SKILL.md` standard and nothing else. Do not use it for tools or
extensions.

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

## Logging

`src/midge/logs.py` owns configuration; every other module only ever acquires a logger.

- `_logger = logging.getLogger(__name__)` at module top. Never `print()`, never a facade, adapter, or wrapper — `getLogger(__name__)` is what makes `%(name)s`, per-module levels, and `caplog` work, and all three break the moment something is put in front of it.
- **Only entrypoints call `configure()`.** Library code never configures logging, because the right handler depends on the mode and only the entrypoint knows it. Stdout is the protocol in RPC mode and the transcript in headless mode, so nothing may write to it. In the TUI, `logging.StreamHandler` binds `sys.stderr` at construction and so writes straight past Textual's `redirect_stderr` and corrupts the display — use `tui_log_handler()`.
- Lazy `%s` arguments, never an f-string in the format string. Enforced by ruff `G`/`LOG`.
- **The first token is a `snake_case` event identity, then `key=%s` pairs.** `_logger.warning("skill_description_missing path=%s", path)`, not `"Skipping skill %s: description is required"`. Errors have to be countable with `grep -c`, not a regex over English.
- Levels: **ERROR** the operation failed · **WARNING** degraded but continuing · **INFO** the operational narrative · **DEBUG** why it did that.
- Every `except` that swallows logs, with `exc_info=e` (or `.exception()`) whenever the traceback would otherwise be lost. A bare `type(e).__name__` is rarely enough to act on.
- Arguments are evaluated whether or not the level is on, so keep them O(1). Anything expensive goes through `logs.payload()`, which defers the work into `__str__`.
- **Payloads only at DEBUG, only via `logs.payload()`** — it truncates at `MIDGE_LOG_PAYLOAD_CHARS` (default 2000). Request bodies, tool arguments and results qualify.
- **Credentials are not payload and are excluded at every level.** An `api_key` is never logged — not a prefix, not a length. A `base_url` goes through `logs.provider_host()`, which keeps the hostname and drops userinfo and query string.
- No logging in pure transforms (`session.py`, `tools/__init__.py`) — their failures already raise with real tracebacks.

## Layout

```
src/midge/            # the harness package
src/midge/tools/      # @tool decorator + built-in coding tools
src/midge/extensions.py  # the loader for tool directories
src/midge/skills.py   # SKILL.md discovery + the system-prompt catalogue
src/midge/logs.py     # logging configuration (entrypoints only)
src/midge/hooks.py    # lifecycle events + handler registry
tests/              # pytest tests
examples/           # entrypoints
notes/              # patterns borrowed from pi-mono during reading passes
```

## Style

- No backwards-compatibility scaffolding for code that does not exist yet.
- No comments explaining what code does — only why, when non-obvious.
- Default to no docstrings unless the function is part of a public API.
- Keep abstractions cheap; three similar lines beat a premature helper.
