# Changelog

All notable changes to midge are documented here.

midge uses [Semantic Versioning]. Before 1.0.0, a minor version may introduce breaking changes.

The RPC protocol has its own version number, carried in the `protocol` field of the `ready` frame (and echoed by `get_state`), which changes only when the wire format changes. See [docs/rpc.md](docs/rpc.md).

[Semantic Versioning]: https://semver.org/

## [Unreleased]

## [0.1.0] - unreleased

### Added

- Streaming agent loop against OpenAI's Responses API or any chat-completions server.
- Retries that respect `Retry-After` headers.
- Model registry for using several provider services at once.
- `@tool` decorator turns an async function into a callable tool with auto-generated schema.
- Built-in coding tools: `read`, `ls`, `grep` (read-only), `write`, `edit`, `bash`.
- Extension loader: `.py` files in `--extension-dir` directories add tools, hooks, sub-agents and profiles.
- Agent Skills: `SKILL.md` directory format for domain-specific instructions, listed in the system prompt.
- Sub-agents: `@subagent` creates `spawn_*` tools that run a nested agent with its own prompt and tool subset.
- Profiles: named bundles of system prompt, model, tools and active hooks, selected via `--profile`.
- Hooks for lifecycle events: block or rewrite tool calls, patch results, reshape requests.
- Append-only JSONL session transcripts, enabled by default.
- Post-turn compaction summarizes old turns when history exceeds a token threshold.
- Textual TUI with session management and profile switching.
- TUI asks before running any tool that is not read-only (on by default, `[tui] approve_tools`); RPC never asks.
- JSON-on-stdio RPC protocol for driving the agent from another program.
- RPC mode persists the conversation to the transcript and resumes it on restart.
- The RPC server's first frame is `ready`, carrying the protocol version. The protocol reference lives in `docs/rpc.md`, with a compatibility policy, and is pinned by a golden-frame test.
- Mutating tool calls run serially in order; read-only calls may overlap.
- Missing or rejected API key reported as a sentence before the first prompt, rather than as an error repr.
- Configuration in `.midge/config.toml` (project) and `~/.midge/config.toml` (user), with precedence flag > environment > file > default. See [docs/config.md](docs/config.md).
- `--session`, `--continue` and `--no-session` flags for session control.
- `--compaction-threshold` and `--compaction-keep-recent` flags.
- `midge --version` and `python -m midge` as entrypoints.
- Configurable system prompt via `[agent] system_prompt` in config.
- Optional built-in coding tools via `[tools] builtin = false`.
- The skills catalogue is shown only when some tool can open a file by path.
- Retargeting guide in `docs/retargeting.md`, with tested code blocks.
- Sub-agent events forwarded to RPC clients with correlation metadata.

### Changed

- `py.typed` marker added so type checkers recognize midge as a typed package.

- Package description corrected in `pyproject.toml`.
- CONTRIBUTING.md branch rule corrected to match CLAUDE.md (feature branches from `develop`).
- Single source for the version: `midge.__version__` now reads from pyproject metadata.
- This CHANGELOG added.

[Unreleased]: https://github.com/kynetyk-ai/midge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kynetyk-ai/midge/releases/tag/v0.1.0
