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
| A nested agent the model delegates to | `src/midge/subagents.py`, `@subagent` → a `spawn_*` tool | **no equivalent** (an example extension only) |

The word **skill** means the `SKILL.md` standard and nothing else. Do not use it for tools or
extensions.

A **subagent** is delegation — the model calls a `spawn_<name>` tool that runs a nested agent and
returns its result. Do not confuse it with supervision (an external orchestrator driving several
midge *processes*), which was considered and declined in #32. Sub-agents are declared in `.py`
files where the function signature is the tool schema and the return value is the child's opening
message, so the model chooses inputs but never the child's prompt, tools, or model.

## Tooling and conventions

- **Poetry** for env and dependency management (`poetry install`, `poetry run <cmd>`, `poetry add <pkg>`). Never `pip`, `uv`, `pip-tools`, or `hatch`. `poetry.lock` is committed.
- **Python 3.11+**. Use `asyncio.TaskGroup`, exception groups, and modern type hints.
- **Providers live in `src/midge/providers/`.** A provider owns one wire format — `encode` / `open` / `decode` / `is_retryable` — and nothing else. The streaming state machine and the retry policy stay in `client.py`, written once; `Delta` is the normalization point. Two names are registered against the OpenAI adapter today (`openai`, `openai-compatible`) because they share a format and differ only in declared `Capabilities`. Do not introduce LangChain or LiteLLM without checking with the user first — the harness loop is small enough that they add weight without buying anything.
- **Pydantic v2** for tool-arg schemas.
- **Textual** for the TUI.
- **Lint:** `ruff`. **Type-check:** `pyright`. **Test:** `pytest` + `pytest-asyncio`.
- **Size budget:** the core harness (`src/midge/*.py`, excluding `tools/` and `tui/`) is capped at 5000 lines, counting neither blanks, comments, nor docstrings. Enforced by `scripts/loc.py` in CI and in `tests/test_loc_budget.py`. Prose is deliberately free — if a change breaches the cap, simplify the code or move what is not core into `tools/` or an extension. Never comply by deleting explanation.

## Out of scope (do not propose without checking)

- ~~Multi-provider zoo beyond `openai`+`base_url`~~ — **reversed.** `src/midge/providers/` now has a protocol and a registry, so a second wire format is a declaration rather than an `if`. Adding one is in scope; the constraint is that it must fit the existing contract without the core learning about it. A new adapter needs a real reason, not symmetry.
- OAuth flows
- Faithful port of `pi-tui`
- `pi-mom` (Slack), `pi-pods`, `pi-web-ui`
- WASM / native deps
- LangChain / LiteLLM (see above)

## Configuration

`src/midge/config.py` owns it. **Anything a user or operator might reasonably want to change goes
there — extend `Config` rather than inventing a knob locally.** That is the default, not a nicety:

- **Never hardcode a tunable.** A magic number, timeout, threshold, path, level, limit or model
  id sitting in a module is a config key that was not filed. If a value would plausibly differ
  between two people, two machines, or two deployments, it belongs on `Config` with a default —
  not in a constant halfway down a module, and never as a literal at the call site. A library
  default in a signature (`Client(max_attempts=3)`) is fine and often necessary, but it is not a
  substitute: the entrypoint must still pass the configured value, or nobody can reach it.
- **Never add an environment variable.** `os.getenv` in a new module is the smell this section
  exists to prevent. Add the field to `Config` and give it an env name *there*, so it appears
  alongside every other setting and in `examples/config.toml`.
- **Extending is cheap and expected.** Three edits: the field on `Config`, its line in
  `Config.load`, and its commented entry in `examples/config.toml`. Do not weigh "is this worth a
  config key?" — a scattered knob costs far more than a field.
- **The reason is discoverability.** Eight environment variables read in ten places were
  findable only by grepping the source. One file with the defaults visible is the whole point;
  every value that escapes back into a module erodes it.

The mechanics, which follow from the same rule as logging below:

- **Only entrypoints construct a `Config`.** It is passed inward; library modules take
  parameters. There is deliberately **no `get_config()`** — a global would let any module reach
  configuration, which is the coupling this module exists to remove, and it would be untestable
  without monkeypatching.
- **No module reads the environment for a *setting*.** `os.getenv` appears in exactly three
  places, and two of them are credentials: the resolver in `config.py`, the `OPENAI_API_KEY`
  fallback in `providers/openai_compat.py`, and `api_key_env` in `providers/registry.py`. The
  rule is the category, not the count — **credentials come from the environment, by name, and
  never from the file; settings never come from the environment outside `config.py`.** A fourth
  read for anything that is not a credential is the thing to push back on.
- Precedence is **flag > env > config > default**. `Config.load` resolves the last three; flags
  win at the call site. An argparse `default=` that is not `None` silently makes the config layer
  unreachable — the default belongs in `Config`, once.
- **`load` parses and logs nothing**, returning `Diagnostic`s the entrypoint passes to `emit`
  after `logs.configure`. It has to run before logging exists, because the log level is one of
  the things it resolves.
- **Bad input is a diagnostic, never an exception.** A malformed file, an unknown key or a
  wrongly typed value degrades to the default and says so. A typo must not stop the harness from
  starting, and must not silently change what it does either.
- **A credential is never a config field.** `OPENAI_API_KEY` stays in the environment; a config
  file gets committed. `api_key` in the file is reported as an unknown key. A second provider
  names *which variable* holds its key (`api_key_env`), never the key.
- **The model registry is the user's, and midge never populates it.** `[models.*]` maps a model
  id to a `[providers.*]` entry, and `providers/registry.py` resolves it per request. Do not ship
  a list of models, do not validate a model id against one, and do not add a "did you mean" —
  vendors churn models continuously and a typo fails at the API with a better error than we could
  give. Validate the *wiring* (a model naming an undefined provider) and nothing else. An empty
  registry is permissive; writing a `[models]` table is what turns enforcement on.
- `tests/test_config.py` asserts `examples/config.toml` still parses with no diagnostics, so
  documenting a new key is not optional — a field added without its example entry fails CI.

## Logging

`src/midge/logs.py` owns configuration; every other module only ever acquires a logger.

- `_logger = logging.getLogger(__name__)` at module top. Never `print()`, never a facade, adapter, or wrapper — `getLogger(__name__)` is what makes `%(name)s`, per-module levels, and `caplog` work, and all three break the moment something is put in front of it.
- **Only entrypoints call `configure()`.** Library code never configures logging, because the right handler depends on the mode and only the entrypoint knows it. Stdout is the protocol in RPC mode and the transcript in headless mode, so nothing may write to it. In the TUI, `logging.StreamHandler` binds `sys.stderr` at construction and so writes straight past Textual's `redirect_stderr` and corrupts the display — use `tui_log_handler(log_file)`.
- Lazy `%s` arguments, never an f-string in the format string. Enforced by ruff `G`/`LOG`.
- **The first token is a `snake_case` event identity, then `key=%s` pairs.** `_logger.warning("skill_description_missing path=%s", path)`, not `"Skipping skill %s: description is required"`. Errors have to be countable with `grep -c`, not a regex over English.
- Levels: **ERROR** the operation failed · **WARNING** degraded but continuing · **INFO** the operational narrative · **DEBUG** why it did that.
- Every `except` that swallows logs, with `exc_info=e` (or `.exception()`) whenever the traceback would otherwise be lost. A bare `type(e).__name__` is rarely enough to act on.
- Arguments are evaluated whether or not the level is on, so keep them O(1). Anything expensive goes through `logs.payload()`, which defers the work into `__str__`.
- **Payloads only at DEBUG, only via `logs.payload()`** — it truncates at `LogConfig.payload_chars` (default 2000, `MIDGE_LOG_PAYLOAD_CHARS`). Request bodies, tool arguments and results qualify.
- **Credentials are not payload and are excluded at every level.** An `api_key` is never logged — not a prefix, not a length. A `base_url` goes through `logs.provider_host()`, which keeps the hostname and drops userinfo and query string.
- No logging in pure transforms (`session.py`, `tools/__init__.py`) — their failures already raise with real tracebacks.

## Layout

```
src/midge/            # the harness package
src/midge/tools/      # @tool decorator + built-in coding tools
src/midge/extensions.py  # the loader for tool directories
src/midge/skills.py   # SKILL.md discovery + the system-prompt catalogue
src/midge/subagents.py # @subagent → spawn_* tools that run nested agents
src/midge/config.py   # .midge/config.toml → Config (entrypoints only)
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
