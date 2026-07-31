# midge

A hackable Python agent harness originally ported from [`pi-mono`](https://github.com/badlogic/pi-mono) (TypeScript) for personal preference, readability, and domain-adaptability.

The core harness is about 3.5k lines of code — blank lines, comments and docstrings excluded — and **CI holds it under 5k** (`scripts/loc.py`). Counting the built-in tools, provider adapters and TUI alongside it, the whole package is roughly 4.5k. The agent loop, OpenAI-compatible client, tool registry, extension loader, RPC server, JSONL session save/load, context compaction, and Textual TUI are all small enough to read in a sitting and modify with confidence.

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
- **JSONL session save/resume, on by default.** Every run records a transcript under `.midge/sessions/` unless you say otherwise. The format is append-only and documented: a rename or a context clear is a record appended and replayed on load, never a rewrite, so a crash can only ever damage the final line. A session spanning several files — a sub-agent writes its own — says so in both directions, so the whole run is walkable from any one of them. Anything that wants to view or watch a session reads the transcript directly.
- **Context compaction** that summarizes old turns when a session gets long.
- **Provider retry** with a cancellable, jittered backoff — rate limits, 5xx, and transport failures get a few attempts before the turn fails. When a 429 says `Retry-After`, that is honoured instead of the backoff, capped so a server asking for an hour cannot park a turn for one. A limit one request hits is shared with every other request on that provider, so concurrent sub-agents do not each spend a rejection learning the same thing. Retries stop once the response has started streaming, so nothing the model already emitted is replayed.
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
poetry run python -m examples.coding_agent --session run.jsonl "summarize the README"
```

`model` defaults to `gpt-4o-mini` — see [Configuration](#configuration). Other flags: `--extension-dir DIR`, `--skill-dir DIR`, `--skill NAME`, `--session PATH` (resumes if the file exists), `--no-session`, `--compaction-threshold N`.

Transcripts are written whether or not you ask. Without `--session`, a timestamped file appears in `.midge/sessions/` (already covered by `.gitignore`); `--session run.jsonl` names one in that same directory, and an absolute path writes wherever you point it. `--no-session` records nothing for one run, and `[session] enabled = false` turns it off for good.

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

Newline-delimited JSON. `get_commands` enumerates everything invocable — built-in commands and `SKILL.md` skills alike — each with a JSON Schema for its arguments, so a client can build a command palette without hardcoding the protocol. `reload` re-scans skills and extensions from disk, so a new `SKILL.md` or an edited tool takes effect without restarting. `open_session` attaches a running agent to another transcript, creating it if the path is free, which is what lets a client leave a conversation and come back to it, and `list_sessions` says which transcripts exist so a client can offer the choice — sub-agent runs and profile excursions excluded, since reopening one would resume the middle of a tool call. The wire format is documented at the top of [`src/midge/rpc/__init__.py`](./src/midge/rpc/__init__.py), and what a transport other than stdio would have to decide for itself at the top of [`src/midge/rpc/transport.py`](./src/midge/rpc/transport.py); [`notes/rpc.md`](./notes/rpc.md) is the port-era reading note rather than current documentation.

**midge never listens on anything** — no socket, no port, no bind address; the only network traffic is outbound to the provider. Stdin and stdout are a capability handed to the process by whoever launched it, so access control comes from the OS and the container runtime rather than from code midge would have to get right. Bridging to a socket is left to whoever deploys it, because the right shape is the client's to decide — and because anything that can send a line can run `bash` with the process's privileges.

**A resumed transcript restores the conversation, not your configuration.** History and the base system prompt come back — resuming a reviewer's session under a coding assistant's instructions would make its own history misleading. The model does not: it is infrastructure with its own config key, so a recorded one beats a default and loses to a model you asked for this run, and a disagreement is reported rather than silently resolved. That also means a session recorded against a model the vendor has since retired warns and falls back instead of refusing to start.

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

Only the child's final answer reaches the parent's **context**, which is the point: a search that
reads thirty files costs the main conversation one paragraph. Its **activity** is still visible —
over RPC, a nested agent's tool executions carry an `agent` envelope naming which run produced
them, so a client can show what a ninety-second delegation is doing rather than watching a spinner.
The envelope is a sibling key (`agent_id`, `parent_id`, `depth`) and is absent on top-level events,
so a client that ignores it sees the stream it always saw. `agent_id` is the id of the spawning
tool call — the same id the child's transcript records — so the wire and the session name a run
identically. The child's own turns go to a sibling transcript
named for the tool call that spawned it, so the delegated work stays inspectable. The link is
recorded both ways — the child's header carries `origin: "subagent"` and a pointer to the parent,
and the parent appends a `continued` record naming the child — so "which transcripts belong to
this run" is a walk from either end rather than a directory scan.

A sub-agent inherits the parent's **tool policy** — an approval hook that blocks a command blocks it
when delegated too — but not the parent's prompt- or request-shaping hooks, which would otherwise
silently override the child's own system prompt.

**Recursion is the allowlist's business.** A child only gets `spawn_*` tools if its own `tools`
names them, so an author who grants nesting has done it deliberately and a global depth cap would
only override a declaration sitting in their file.

What must not happen is recursion with no end, and that is denied precisely rather than bounded by
a number: **a child never receives a spawn tool for an agent already running above it.** So
`alpha → beta` always works, and `beta → alpha` is refused only where alpha is on the stack —
called from anywhere else the same declaration is fine. Nothing is dropped and no declaration is
edited, and termination follows anyway, since the ancestor set grows by one name per level from a
finite declared set.

A cyclic allowlist is still a bug worth knowing about, so it's reported at startup — along with a
tool name that resolves to nothing. Neither costs you the agent. What does is a **model no
`[models]` entry names**, which is dropped, because unlike a typo that agent cannot run at all —
it would otherwise surface as the vendor's 404 inside a turn.

A delegation is always bounded, by three people. The author sets a budget per agent
(`timeout=180`); a `spawn_*` tool may offer the *caller* a `timeout` parameter by declaring one in
its signature, for a job that needs longer; and `[subagents] max_timeout` caps the lot, so
offering the knob never means offering none.

See `examples/subagent_extension/` and [`notes/subagents.md`](./notes/subagents.md).

### Profiles — what the agent *is*

A sub-agent is a tool the agent uses; a **profile** is the agent itself. It bundles a system
prompt, a model, a subset of the discovered tools and a set of active hooks under one name, so
that "the adversarial reviewer" is a thing rather than three unrelated changes:

```python
from midge.profiles import Profile

ADVERSARIAL = Profile(
    name="adversarial-reviewer",
    description="Reviews work that has just been done, looking for what is wrong with it.",
    tools=("read", "bash"),      # read-only: a reviewer that can edit fixes instead of reporting
    hooks={"approve": True},     # a decision for every hook source, keyed by file stem
    prompt="Assume the work is wrong and find out how. Cite `path:line` for every claim.",
)
```

A profile lives in an ordinary extension file — one `.py` may declare a tool, a sub-agent and a
profile together — so `--extension-dir` discovers it and there is no separate flag. Pick the one
to start under with `--profile NAME` or `[profiles] default` in the config file.

Validation is strict where the rest of the harness is lenient: a profile naming a tool, a hook, or
(with a `[models]` table configured) a model that does not exist is **dropped with a diagnostic**
rather than loaded, so it can never silently grant less than it claims.

`hooks` is stricter still — it must decide **every** discovered hook source, and a profile that
leaves one out does not load. The asymmetry with `tools` is deliberate: omitting a tool yields a
less capable agent, while omitting a hook would yield an unguarded one. Requiring a decision per
source means disabling an approval gate is something you write, never something you forget.

```bash
poetry run midge --extension-dir examples/profile_extension \
  --extension-dir examples/approval_extension --profile adversarial-reviewer
```

`--profile` starts under one. Over RPC, `use_profile` switches at runtime — **every dimension or
none**, refused mid-turn, and recorded, so any message is attributable to the configuration that
produced it. That it is one operation is the point: a client hand-orchestrating a switch gets
`success: true` from `set_system_prompt` while the entire previous toolset and every hook stay
active, and no client discipline fixes that.

`transcript` says where the turns go, and the three values exist because the round trip needs
them:

```jsonc
{"type":"use_profile","name":"builder"}                              // continue
{"type":"use_profile","name":"reviewer","transcript":"fork"}         // excursion
{"type":"use_profile","name":"builder","transcript":"resume_last"}   // and back
```

`fork` opens a linked transcript so a review's turns do not land in the build thread's file;
`resume_last` returns to this session's most recent thread under the named profile, which is what
makes the excursion a round trip rather than a one-way door. It walks the session's own transcript
chain — never a directory scan, never another session — and excludes sub-agent runs by their
`origin`. A profile used for the first time has nothing to resume, which is an ordinary first run:
the switch still succeeds, falling back per `[profiles] resume_fallback` (default `fork`), and the
response says which happened.

**There is no revert.** Going back is naming the other profile, so the operation stays stateless
and a client never reasons about how deep it is in a sequence of excursions. **History is
untouched** — `fork` changes which file the turns are written to, not what the agent still holds;
compose `clear_context` after if you want a clean slate.

Resuming a session brings its profile back, and warns rather than refusing if that profile is no
longer discovered. See [`docs/adr/0001-session-profiles.md`](./docs/adr/0001-session-profiles.md)
for the design and what was deliberately rejected along with it.

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
├── rpc/               # JSON-on-stdio front-end: wire / server / transport
├── extensions.py      # load_extensions(dirs) → (ToolRegistry, prompt_addition)
├── config.py          # .midge/config.toml → a Config the entrypoint passes inward
├── logs.py            # logging config; entrypoints only
├── skills.py          # SKILL.md discovery + <available_skills> catalogue
├── subagents.py       # @subagent → spawn_* tools running nested agents
├── profiles.py        # Profile discovery + validation (what the agent *is*)
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
├── profile_extension/ # a declared profile: the adversarial reviewer
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
8. Anything else, in any order: `compaction.py`, `persistence.py`, `rpc/`, `tui/app.py`.

## License

MIT — see [`LICENSE`](./LICENSE). Copyright (c) 2026 Kynetyk Holdings LLC.

Lineage and dependency credits in [`ACKNOWLEDGEMENTS.md`](./ACKNOWLEDGEMENTS.md).
