# Skills — patterns borrowed from `pi-mono`

Source:
- `pi-mono/packages/coding-agent/src/core/skills.ts` — discovery, validation, prompt formatting
- `pi-mono/packages/coding-agent/src/core/system-prompt.ts:63-67` — the read-tool gate
- `pi-mono/packages/coding-agent/src/core/agent-session.ts:1297-1325` — `/skill:name` expansion
- `pi-mono/packages/coding-agent/docs/skills.md` — the user-facing contract
- The published standard: <https://agentskills.io/specification>

Implemented in `src/midge/skills.py`. See `notes/extensions.md` for the sibling
subsystem — extensions add *tools*, skills add *instructions*, and they share
nothing but the fact that both concatenate into the system prompt.

## The one decision that makes this cheap: there is no skill tool

pi puts a catalogue of name, description and **absolute path** into the system
prompt and tells the model to open the file with the `read` tool it already has.
`<location>` is the entire mechanism. "Invoke a skill" reduces to "call `read`
on a path" — no new tool, no new message type, no agent-loop change.

The `Skill` record holds no body. That single fact is what makes progressive
disclosure fall out for free instead of needing machinery: only descriptions are
ever resident, and the instructions load when the model decides they are
relevant.

Bundled `scripts/`, `references/` and `assets/` need no support at all. They are
covered by one sentence in the prompt telling the model to resolve relative
paths against the directory containing `SKILL.md`. The model constructs the
absolute paths itself and runs them through `read` and `bash`.

## Compaction needs no changes

The *durable* part (name, description, path) lives in `system_prompt`, which is
a separate field on `Agent` and is not part of `history`. The *ephemeral* part
(the body, once read) is disposable and re-fetchable from the path. So the
positional cut in `compaction.py:189-211` can summarize a loaded body away
without breaking anything — the catalogue that tells the model how to get it
back is untouched.

pi confirms this by omission: zero occurrences of "skill" across both of its
compaction implementations. There is no pin/protect/preserve concept.

A body pulled in by `skill_message` *is* an ordinary user message and is
therefore compactable. That is the intended trade-off, not an oversight.

## Leniency is a feature, not sloppiness

Everything except a missing description is a warning that still loads:
over-long name, wrong charset, leading/trailing/consecutive hyphens, over-long
description. A missing description is the one hard failure, because the
description is the only thing the model ever sees up front — a skill without one
is unreachable by construction.

pi went further and dropped the spec's requirement that `name` match the parent
directory (`docs/skills.md:7`), because that rule is hostile to skill
directories shared between harnesses. midge does the same. This is what lets
`--skill-dir ~/.claude/skills` work.

Any exception is a diagnostic and a skip. A malformed skill must never abort
startup.

## Discovery rules worth keeping

- **A directory containing `SKILL.md` is a leaf** — stop recursing. This is what
  keeps a bundled `references/` tree from being read as a nest of sub-skills,
  and it makes nested sub-skills structurally impossible.
- Skip dotfiles, `node_modules`, `__pycache__`.
- De-duplicate by resolved path *before* the name check, so the same file
  reached twice is silent rather than a false collision.
- Name collisions: first wins, with a diagnostic naming winner and loser.
- **Precedence is a property of the ordered source list, not of the order the
  walk happened to reach files in.** pi got this wrong once and fixed it with a
  rank function plus a regression test.

Python simplifies two things pi does by hand: `Path.is_file()` follows symlinks
and answers `False` for a broken one, so no explicit stat dance is needed; and
`xml.sax.saxutils.escape` replaces its five-call `.replace` chain.

## Departures from pi

| pi | midge | why |
|---|---|---|
| Tiered discovery: global, project, ancestor walk to the git root, package `skills/` entries, settings array, trust gating | `--skill-dir` plus four defaults from `default_skill_dirs()` | midge has no settings file, no package manager, and no trust model to hang it on |
| `includeDefaults` flag inside the loader | `default_skill_dirs()` is a separate function feeding the source list | pi's flag left a dead branch behind when path collection moved elsewhere; there is nowhere for one to live if there is no branch |
| gitignore / `.ignore` / `.fdignore`-aware scanning | none | needs a dependency, and the leaf rule already stops the walk at the first `SKILL.md` |
| Root-level loose `.md` files count as skills in pi's own config dirs | directories only; pass a file path explicitly to load one | the `includeRootFiles` flag applies at one level only and is exactly the half-live surface worth not having |
| Unbounded recursion | `_MAX_DEPTH = 6` | `--skill-dir ~` should not crawl a home directory |
| Structured `ResourceDiagnostic[]` returned to callers | `logging.warning` on `midge.skills` | pi has a settings selector and a diagnostics pane to consume them; midge has neither, and `load_extensions` already warns-and-continues |
| `disableModelInvocation` (negative) | `model_invocable` (positive) | negative booleans read badly at the one site that checks them |
| `/skill:name` slash command, expanded client-side | `skill_message(skills, name, instructions)` returning a `UserMessage` | midge has no command layer; `Agent.stream` already accepts a `UserMessage`, so forcing needs no new seam |
| Unknown skill in `/skill:` passes through as literal text | `skill_message` raises `KeyError` | pi is inside a text-expansion path where a bare typo should be sent literally; a programmatic caller with a bad name has a bug |

The `<skill name=… location=…>` envelope is kept byte-compatible with pi's, so a
transcript is legible across both harnesses.

## Not implemented, on purpose

`allowed-tools`, `license`, `compatibility` and `metadata` are in the published
spec, are accepted, and are ignored. `allowed-tools` is unimplemented in pi too
— it appears in one docs table and nowhere in the source. If skill-scoped
permissions are ever wanted, the `tool_call` gate in `hooks.py` is the right
place for them, not the skill loader.

## The read-tool gate matters more here than in pi

pi gates the catalogue on a read-capable tool being registered, but in practice
`read` is always there, so the check is near-vestigial. In midge it is load
bearing: goal 3 is retargeting the harness to non-coding domains, and
`examples/notes_agent.py` genuinely runs with no `read` tool. Without the gate
that agent would be instructed to use a tool it does not have.
