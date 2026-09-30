# midge documentation

Start with [architecture](architecture.md) for how the pieces fit and an order for reading the source.

## Subsystems

| Doc | Covers |
|---|---|
| [agent-loop](agent-loop.md) | The turn loop: streaming, tool dispatch, steering, cancellation |
| [providers](providers.md) | The streaming client, retries, provider adapters, the model registry |
| [tools](tools.md) | `@tool`, `read_only`, the built-in coding tools |
| [extensions](extensions.md) | Loading user `.py` files that add tools, hooks, sub-agents and profiles |
| [skills](skills.md) | `SKILL.md` discovery, the catalogue, invoking a skill |
| [sub-agents](subagents.md) | `@subagent` and the `spawn_*` tools it creates |
| [profiles](profiles.md) | Named configurations the agent runs as |
| [hooks](hooks.md) | Lifecycle events and what a handler can change |
| [sessions](sessions.md) | The JSONL transcript: saving, resuming, opening another |
| [compaction](compaction.md) | Summarizing old turns to stay within the context window |
| [config](config.md) | `.midge/config.toml`, precedence, credentials, the model registry tables |
| [logging](logging.md) | Log configuration and the message format |
| [rpc](rpc.md) | The JSON-on-stdio wire contract |

## Guides

- [Running midge](running.md): install, flags, the TUI, containers, Docker Sandboxes, the one-shot CLI.
- [Retargeting](retargeting.md): build a non-coding domain from an empty directory.

## Decisions

- [ADR 0001: session profiles](adr/0001-session-profiles.md)

## Examples

Runnable examples live in [`examples/`](../examples/). Each is referenced from the doc it illustrates.

| Example | Shows |
|---|---|
| [`config.toml`](../examples/config.toml) | Every config key, commented, with its default |
| [`coding_agent.py`](../examples/coding_agent.py) | A one-shot CLI for the coding domain |
| [`rpc_client.py`](../examples/rpc_client.py) | A minimal client for `midge --rpc`, standard library only |
| [`approval_extension/`](../examples/approval_extension/) | A tool-approval hook over a denylist (advisory) |
| [`allowlist_extension/`](../examples/allowlist_extension/) | A restriction that holds: named tools only, writes under cwd |
| [`notes_extension/`](../examples/notes_extension/) | A complete non-coding domain: tools and a profile |
| [`subagent_extension/`](../examples/subagent_extension/) | A read-only explorer sub-agent |
| [`profile_extension/`](../examples/profile_extension/) | A declared profile: an adversarial reviewer |
| [`skills/`](../examples/skills/) | A worked `SKILL.md` with a reference file |
