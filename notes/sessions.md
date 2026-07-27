# Session save/load — patterns to borrow from `pi-mono`

Source: `pi-mono/packages/coding-agent/src/core/session-manager.ts` (and surrounding `core/sessions/`)

## What pi-mono ships and what we don't need

Pi-mono's session format is **far more featureful** than our v1 needs. It supports:

| Feature | Why we skip it for v1 |
|---|---|
| Tree of entries with `parentId` (branching/forking) | We're a linear-history agent. |
| Multiple session files per `cwd` directory (`~/.pi/agent/sessions/<encoded-cwd>/*.jsonl`) | One file per run is enough. |
| 8 entry types: `message`, `thinking_level_change`, `model_change`, `compaction`, `branch_summary`, `custom`, `custom_message`, `label`, `session_info` | We only need `message` and `compaction`. |
| Schema migrations (v1 → v2 → v3) | Brand-new format; one version. Hard fail on mismatch with a clear error. |
| Buffered first-assistant write | An optimization for human-typing-pause scenarios; not needed. |
| UUIDv7 IDs + parent pointers | Our linear history needs no IDs. |
| Tool/model schema NOT persisted (rebuilt from registry on load) | Same — sensible default; we adopt this. |
| Display-name `SessionInfoEntry` records | `name` field on the header is enough. |

This is not aspirational *future* work — it's stuff we'd add only if a real use case shows up. The minimum-viable v1 is dramatically simpler.

## What v1 looks like

**Format:** newline-delimited JSON. One JSON object per line. Trailing newline at EOF.

**Default location:** None. The user passes a path. We don't impose `~/.pi/sessions/...` conventions.

**Entry types (v1):** two.

```jsonl
{"type":"header","version":1,"created_at":"2026-05-02T18:00:00Z","model":"gpt-4o","system_prompt":"You are..."}
{"type":"message","data":{"role":"user","content":"hi","timestamp":1746201600000}}
{"type":"message","data":{"role":"assistant","content":[...],"model":"gpt-4o","stop_reason":"stop","timestamp":1746201601000}}
{"type":"message","data":{"role":"tool_result","tool_call_id":"c1","tool_name":"read","content":[...],"is_error":false,"timestamp":...}}
{"type":"compaction","summary":"## Goal ...","cut_index":12,"timestamp":1746201700000}
```

The `data` for a `message` entry is exactly `Message.model_dump(mode="json")` — same shape we already serialize today. No transformation, no field renames. This means session files round-trip cleanly through our existing Pydantic models.

The `compaction` entry is **informational**, not load-bearing. After compaction runs, we *also* append the resulting synthetic user message as a normal `message` entry. The `compaction` entry just records "this happened, here's what the summary text was, this many turns were folded in" — useful for HTML rendering and debugging.

**Append-only.** Each call to `Session.append(message)` writes one line, flushes, returns. No buffering. No rewrite-on-modify. No deletions.

**Header is mandatory and must be the first record.** Loading a file without a header is an error. Loading a file with `version != 1` is an error with a "format incompatible" message — no migration code in v1.

## Lifecycle

```python
# starting a new session
session = Session.new(path, agent=agent)
# (writes the header)

# during the run, hook into agent.history changes
agent.history.append(msg)
session.append(msg)

# resuming
session = Session.load(path)
agent = build_agent_from_header(session.header)
agent.history = session.messages  # full reconstruction
```

For v1 we want this to be **opt-in** — `examples/coding_agent.py` gets a `--session PATH` flag; if given, the agent persists turns to that file (creating it if missing, appending if present). Without the flag, no file I/O.

## What load does *not* do

- Does not validate that the registered tools match what was used in the saved session. The model will try to call tools by name; if a tool is missing, our existing tool-error path handles it (returns `is_error=True` to the model on the next turn).
- Does not validate that the model name matches. If it changed, the user knows.
- Does not re-run anything. Loading is a passive reconstruction.

## What gets persisted vs. derived

| Persisted | Derived on load |
|---|---|
| Header: `version`, `created_at`, `model`, `system_prompt` | The `Client` (constructed fresh from env) |
| Each `Message` verbatim | `agent.history` (rebuilt from the `message` entries in order) |
| `compaction` entry (informational) | The synthetic summary message itself is persisted as a normal `message` entry, so it's already in the reconstructed history |

The `system_prompt` is in the header so a session loaded by a different process gets the same agent persona. The `tools_registry` is **not** persisted — the loader uses whatever extensions the current process loads (per `--extension-dir`). This matches pi-mono's behavior and is the right call.

## Concurrency

A single Python process at a time owns a session file. We don't add locks for multi-writer scenarios. Mention in docstring: "Don't open the same session file in two processes; results are undefined."

## Translation guide

| TS pattern | Python equivalent |
|---|---|
| `appendFileSync(path, line)` | Open in `"a"` mode at session creation; `f.write(line); f.flush()` per record |
| `uuidv7()` IDs | None — linear history doesn't need them |
| `parentId` tree walks on load | Not applicable; load is just `[json.loads(line) for line in file]` |
| Migrations (`migrateV1ToV2`) | Not applicable; hard fail with clear error |
| Buffered first-assistant write | Skip; flush every record |

## What Phase 3 implements

A new module `src/midge/persistence.py` (we already have `src/midge/session.py` for HTML export — calling this one `persistence.py` keeps the names from colliding):

```python
class Session:
    @classmethod
    def new(cls, path: Path, *, model: str, system_prompt: str | None) -> Session: ...

    @classmethod
    def load(cls, path: Path) -> Session: ...

    def append(self, message: Message) -> None: ...
    def append_compaction(self, *, summary: str, cut_index: int) -> None: ...
    def close(self) -> None: ...

    @property
    def messages(self) -> list[Message]: ...

    @property
    def header(self) -> SessionHeader: ...
```

Plus integration: `Agent` grows an optional `session: Session | None = None` arg; if set, every history append also goes to the session. (Or `examples/coding_agent.py` does the wiring at the call site — simpler, lets us not touch `Agent` for now.)

Tests:
- Round trip: new → append several messages → close → load → messages match.
- Load rejects file without header.
- Load rejects file with `version != 1`.
- Multiple message types (user with str, user with list, assistant with text + tool calls, tool_result with error) all round-trip via `Message.model_validate`.
- Compaction entry persists alongside messages.
- Append after load works (we can resume a session).

`examples/coding_agent.py` gains `--session PATH`. If the path doesn't exist, create new; if it exists, load + resume.
