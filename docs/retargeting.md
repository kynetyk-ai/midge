# Retargeting midge to a new domain

midge ships as a coding agent, but nothing in its core is about code. A domain is three things you
write and two lines of config you set:

| You write | What it decides |
|---|---|
| **Tools** — `@tool` functions in a `.py` file | What the agent can *do* |
| **A profile** — a `Profile` in the same file | What the agent *is*: its identity, and which tools it may use |
| **Skills** (optional) — `SKILL.md` directories | *How* to do particular tasks, loaded when relevant |

| You set | Why |
|---|---|
| `[tools] builtin` | Whether midge's coding tools load at all |
| `[profiles] default` (or `--profile`) | Which profile the agent starts as |

The notes domain in `examples/notes_extension/notes.py` is a complete worked example, and runs as:

```bash
midge --extension-dir examples/notes_extension --profile notes
```

This guide builds a smaller one from an empty directory: a **reading list**, an agent that tracks
what you mean to read and what you thought of it.

## 1. An empty directory

```bash
mkdir -p reading/.midge reading/extensions
cd reading
```

midge reads `./.midge/config.toml` in the directory it starts in, and loads extensions from any
directory you name. Everything below lives here.

## 2. Tools

A tool is an `async` function with type hints and a docstring. The signature *is* the schema the
model sees; the docstring is its description. `extensions/reading.py`:

```python
import json
from pathlib import Path

from midge.tools import tool

SHELF = Path(".midge/shelf.json")


def _load() -> dict[str, dict]:
    return json.loads(SHELF.read_text()) if SHELF.exists() else {}


@tool
async def add_book(title: str, author: str) -> str:
    """Add a book to the reading list."""
    shelf = _load()
    shelf[title] = {"author": author, "status": "to read", "notes": ""}
    SHELF.write_text(json.dumps(shelf, indent=2))
    return f"Added {title!r}."


@tool
async def finish_book(title: str, thoughts: str) -> str:
    """Mark a book read, with what you thought of it."""
    shelf = _load()
    if title not in shelf:
        raise KeyError(f"No book titled {title!r}")  # the model is told this, verbatim
    shelf[title] |= {"status": "read", "notes": thoughts}
    SHELF.write_text(json.dumps(shelf, indent=2))
    return f"Finished {title!r}."


@tool(read_only=True)
async def list_books(status: str | None = None) -> str:
    """List the reading list, optionally only 'to read' or 'read'."""
    rows = [f"{t} — {b['author']} ({b['status']})" for t, b in _load().items()
            if status is None or b["status"] == status]
    return "\n".join(rows) or "The list is empty."
```

Two declarations are worth getting right, because midge acts on them:

- **`read_only=True`** says the tool only looks. Read-only calls in one model response may run
  together; anything else runs alone, in the order the model asked. In the TUI, anything *not*
  read-only is shown to the person for approval first (`[tui] approve_tools`). Leave it `False`
  unless it is true — the default is the safe one.
- **`reads_paths=True`** says the tool takes a file path and returns the file's text. It is what
  lets midge offer the model a catalogue of skills (step 5). `list_books` does not qualify — it
  reads a *title*, not a path.

An exception a tool raises is reported to the model as that tool's error, message intact, so raise
with a sentence the model can act on.

## 3. The identity: a profile

In the same file:

```python
from midge.profiles import Profile

READING = Profile(
    name="reading",
    description="A reading-list companion.",
    tools=("add_book", "finish_book", "list_books"),
    hooks={},
    prompt=(
        "You keep the user's reading list. Before adding a book, check the list "
        "so nothing is added twice. When they finish one, ask what they thought."
    ),
)
```

- **`prompt` replaces the base prompt.** midge still appends what extensions add
  (a module-level `SYSTEM_PROMPT` string) and the skills catalogue.
- **`tools` is an allowlist.** Anything not named is unavailable while this profile is active —
  including midge's coding tools, if they are loaded.
- **`hooks` must decide every hook source that is loaded**, by file stem: `{"approve": True}`.
  This extension registers none, so `{}`. A profile that leaves one undecided is dropped with a
  warning rather than guessed at.
- `model` is optional; empty means "keep whatever model is configured".

A profile is validated at startup against the tools, hooks and models that exist, so a typo is a
warning at launch, not a confused model later.

## 4. Drop the coding tools

`.midge/config.toml`:

```toml
[tools]
builtin = false          # or a list: ["read"] keeps just that one

[extensions]
enabled = true
dir = "extensions"

[profiles]
default = "reading"
```

With `builtin = false` midge's `read`, `ls`, `grep`, `write`, `edit` and `bash` are never
imported, so your tools can use those names if they want them. With the profile active they would
be unavailable anyway; `builtin` is about whether they exist at all.

`[extensions] enabled` loads the directory without a flag. It is off by default because an
extension is arbitrary Python imported at startup — turning it on in a project is a statement that
you trust that project's code.

If you would rather not use a profile at all, `[agent] system_prompt` (or `system_prompt_file`)
sets the base prompt directly — the identity any session starts with when no profile is active.

## 5. Skills (optional)

A skill is a directory with a `SKILL.md` — instructions for one kind of task, in the
[Agent Skills](https://agentskills.io/specification) format. Put them in `.agents/skills/` or
`.midge/skills/`, or name a directory with `--skill-dir`.

How the model reaches one depends on whether any active tool declares `reads_paths`:

- **With a path reader**, midge adds a catalogue to the system prompt — each skill's name,
  description and file path — and tells the model to open the file with that tool when a task
  matches. `[tools] builtin = ["read"]` is the simplest way to have one.
- **Without one**, there is no catalogue: offering the model files it cannot open is worse than
  saying nothing. Skills are still usable by name — `/skill:<name>` in the TUI, or a `prompt`
  starting `/skill:<name>` over RPC — because midge reads the file itself and puts it in the turn.

## 6. Run it

```bash
midge                                   # the TUI, as the `reading` profile
midge --rpc                             # the same agent over JSON-on-stdio
python examples/rpc_client.py --server "midge --rpc" "what's on my list?"
```

Everything midge does for the coding agent it does for this one: sessions saved and resumable
(`--continue`), compaction, the approval prompt for `add_book` and `finish_book` in the TUI (and
not for `list_books`), sub-agents, and the RPC protocol in [`docs/rpc.md`](rpc.md).

## 7. What the core still decides

These stay midge's, the same for every domain, because they are how an agent *runs* rather than
what it is:

- **The loop** — streaming, retries, the tool-call ordering rule, and repairing history a provider
  would refuse.
- **Persistence** — the transcript format, and what a resume restores (the conversation and its
  recorded identity, not your current config).
- **Compaction** — when and how old turns are summarized.
- **The front-ends** — the TUI's layout and approval prompt, and the RPC protocol.
- **Trust** — midge runs with its process's privileges. Over RPC nothing asks before a tool runs;
  the boundary is the container or user it runs as.

## A note on settings

An extension has no `Config` of its own yet. The notes example reads one setting from the
environment (`MIDGE_NOTES_KB`, where its knowledge base lives); the reading list above uses a
path relative to the project. midge's own settings never come from ad-hoc environment variables —
they are all in `Config` — and giving extensions the same would be the next step if a domain needs
more than a path or two.
