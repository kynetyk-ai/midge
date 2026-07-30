# ADR 0001 — Session profiles, and rejecting the session tree

- **Status:** Proposed
- **Date:** 2026-07-30
- **Supersedes:** the forking half of #49
- **Related:** #44 (sub-agents), #46/#54 (reload), #49, #55 (session markers), #57 (durability bug)

## Why this document exists

The question "can midge switch the agent's identity mid-conversation?" was
followed through three framings before it settled. First it looked like a
feature. Then like forking. Then like `pi`'s session tree, which is a storage
redesign. It is none of those.

The reasoning ended up spread across issue comments, and issue comments are a bad
place for it — the next person to read #49 would have found a plan for an entry
tree with parent pointers and built the wrong thing. This records what the
concept actually is, what already exists to build it from, and what is
deliberately being refused.

## What is actually wanted

> Make the agent do something different, then be able to revert back.

The motivating case: an agent has just written a feature. You want it to review
that work adversarially — different instructions, a narrower toolset, possibly a
different model — and then go back to being the thing that was building.

That is not a fork of a *conversation*. It is a swap of the agent's
*configuration*. Sharing the word "fork" with `pi` was doing real damage to the
design, because it dragged in a data model built to answer a different question.

## Decision 1 — There is a concept called a profile

A **profile** is:

| dimension | what it sets |
|---|---|
| system prompt | the durable base half; midge still appends the generated half |
| model | the provider model id used for subsequent turns |
| tools | which of the discovered tools this agent can see |
| hooks | which registered hook handlers are active |

Not part of a profile: session file, history, skills catalogue. A profile is a
kind of *agent*; a session is a *conversation*. The same profile should be usable
across many conversations.

## Decision 2 — Reject the entry tree

`pi` gives every session entry an `id` and a `parentId`, so the file is a tree;
a leaf record names the current tip, and loading walks from leaf to root
(`packages/agent/src/harness/session/session.ts`). That supports forking at an
arbitrary past entry, branch switching inside one file, and `get_tree`.

**midge will not have this.** No entry ids, no `parentId` on messages, no leaf
pointer, no `get_tree`. `cut_index` stays a positional offset.

This is a rejection, not a deferral, and it is only sound because of what follows
from Decision 1: **nothing in midge forks at an arbitrary past point.** Those two
things are separable in a way that was not obvious at first —

| what a new context starts from | needs a cut point? |
|---|---|
| nothing | no |
| a summary generated at the time | no |
| the parent's current context, verbatim | **no** — "here", not a position |
| the first N messages of the parent | **yes** |

Only the last requires naming a position, and it is the only one that forces
entry ids. Its use case is "rewind past three bad turns and land exactly on turn
eight", which in practice is served by steering the agent instead. Giving that up
buys the removal of an entire storage redesign.

The cost, stated plainly: you cannot return to an exact earlier conversation
state. You can start clean, start from a summary, or carry the whole current
context.

## Decision 3 — Most of the machinery already exists

This was the surprise, and it is why this is a small feature rather than a large
one.

| dimension | mechanism today | status |
|---|---|---|
| system prompt | `set_system_prompt` sets the base and recomposes, so the skills catalogue and extension contributions survive | works in-process; **not durable** (#57) |
| model | `set_model` | works in-process; **not durable** (#57) |
| tools | `_child_registry` in `subagents.py` projects a subset `ToolRegistry` from an allowlist of names | **exists** |
| hooks | `_Registration.source` is stamped by `_SourceScopedHooks` (`extensions.py`) | **stamped and never read** |
| transcript linkage | `parent_session` / `parent_tool_call_id` on `SessionHeader` | **exists** |

### Tools are filtered by projection, not by discovery

Tools *are* auto-discovered: `load_extensions` walks each source directory,
imports every public `.py`, and collects every `Tool` instance into a fresh
`ToolRegistry` with no filtering. That stays as it is.

Filtering happens one layer up, and sub-agents already do it — `_child_registry`
iterates the parent registry and keeps only names in the spec's allowlist. A
profile uses the same shape: **discover everything once, project a subset per
profile.**

Filtering must *hide* rather than deny-at-call-time. Both are possible — a
`tool_call` hook can refuse a call by name — but `notes/subagents.md` is already
explicit that telling a model it has a tool the registry will reject is
misinformation, not access control. A profile changes what the agent *is*, so the
schemas it receives must match.

### Hooks are the one genuinely new mechanism

There is no way to selectively activate hooks today. `Hooks.clear()` is
all-or-nothing, `on()` returns an unsubscribe closure that `load_extensions`
discards, and while every registration carries a `source`, nothing ever filters
on it — the field exists only to name the offending file in a warning.

Source-scoped activation (`remove_source`, or an enable/disable by source) is
required. This was designed during the reload work (#54) and dropped as premature
because `load_extensions` was the only registrar, so a blunt `clear()` did
identical work. **Profiles are the consumer that makes it not premature.**

## Decision 4 — Switching is atomic or it does not happen

A profile switch must apply every dimension or none, and must be refused while a
turn is in flight (as `reload` already is — swapping a registry mid-turn breaks
tool-call/result pairing).

This is the single strongest argument for making this one operation rather than
documenting a sequence of existing commands. Today a client that tried to
hand-orchestrate a switch would issue `set_system_prompt` plus a session change
and get `success: true` on both — while **the entire previous toolset and every
hook stayed active**, because extension and skill sources are fixed at
`RpcServer.__init__` and `reload` only re-runs the same ones. A partial switch
that reports success is worse than no switch at all, and no amount of client
discipline fixes it, because the capability to change sources is simply absent.

## Decision 5 — Mutable state stays append-only

Restating the invariant #55 established, because a profile switch is the next
thing that will be tempted to break it.

Anything mutable is an appended record replayed at load, last write wins. Never a
header rewrite. Two independent reasons, and they point the same way:

1. **Traceability.** midge is often driven over RPC, where the transcript is the
   only durable account of what happened.
2. **Crash recovery.** `read_transcript` salvages a truncated final line
   (`persistence.py:146`) precisely because nothing earlier is ever rewritten.
   Rewriting the header would trade that away.

So a profile switch is **recorded**, not applied silently — one record naming the
profile, so any message is attributable to the configuration that produced it.
That includes the model, which matters for both cost and behaviour attribution.

### Compaction and clear stay in-thread

An earlier proposal was that compaction and clear should each seal the file and
start a new one, so that one file is exactly one context window. Rejected, and
the reason is worth recording because the motivation turned out to be mistaken.

The motivation was traceability. But both operations are **already
non-destructive**: `append_compaction` and `append_clear` write one record and
collapse only the in-memory view. The file keeps every message, and `cut_index`
says exactly which were folded, so the audit trail is intact. What is lossy is
the *agent's* forward knowledge, which is a property of compaction as a
technique, not of the transcript.

What forking on compaction would actually buy is that an export shows precisely
what the agent saw. That is elegance, not audit — and it would make the TUI's
automatic threshold compaction silently spawn files, and stop `--session PATH`
naming the file being written.

## Open question A — What shape is a profile declared in?

Not settled. A profile is data — a name, a description, a prompt, a model, two
allowlists — and midge has two established discovery idioms to borrow from:

- **Extensions**: `--extension-dir`, walk, import `.py`, collect `Tool`
  instances, first-registered wins with a `tool_name_shadowed` warning.
- **Skills**: `--skill-dir`, walk to depth 6 with a leaf rule, parse `SKILL.md`
  YAML frontmatter, plus `default_skill_dirs()` covering project and home. First
  to claim a name wins, with a `skill_name_shadowed` warning.

### Candidate 1 — a discovered Markdown file with frontmatter (leading)

```markdown
---
name: adversarial-reviewer
description: Reviews recent work looking for what is wrong with it.
model: gpt-4o
tools: [read, bash]
hooks: [approval]
---

You are reviewing work that has just been done. Assume it is wrong and find
out how. Cite `path:line` for every claim...
```

**The body is the system prompt.** That is the property that makes this
attractive: a system prompt is prose, and prose belongs in a file you read rather
than a Python string literal. `_split_frontmatter`, the directory walk, and the
`default_skill_dirs` pattern in `skills.py` are all close to reusable.

### Candidate 2 — a `@profile` decorator in a `.py` file

Discovered by the existing extension loader, exactly as `@subagent` is. Maximum
consistency with sub-agents, and a profile could share prompt text with a
`SubagentSpec` by importing it. Against: a profile is pure data, so this executes
arbitrary code to obtain a dict.

### Candidate 3 — one central config file

No discovery walk, and honest about profiles being few. Against: diverges from
both existing idioms, and a project cannot drop in a profile without editing
shared config.

### Three constraints on whichever wins

- **Vocabulary.** CLAUDE.md is explicit: *"The word skill means the `SKILL.md`
  standard and nothing else."* Borrowing the file *idiom* is fine. Calling a
  profile a skill, extending `SKILL.md`, or implying conformance to the Agent
  Skills specification is not. Candidate 1 would be midge's own format that
  happens to rhyme with one, and should say so.
- **`SubagentSpec` is already a profile** in all but name: `name`, `prompt`,
  `tools`, `model`, plus `timeout`. Either the two converge or this ADR must say
  why they stay apart. The behavioural difference is real — a sub-agent is
  delegation the *model* invokes, a profile is reconfiguration an *operator*
  invokes — but the data is the same, and duplicating it is a smell.
- **Discovery has to earn its keep.** Three profiles may not justify a directory
  walk. The counterweight is enumeration: discovered profiles can be exposed
  through `get_commands` as an enum on `use_profile`, exactly as `reload` exposes
  its `targets`, and then any client renders a profile picker with nothing
  hardcoded.

## Open question B — What happens to the transcript on a switch?

Two intents, which may want different shapes:

- **A revertible excursion.** "Review this adversarially, then put things back."
  The excursion's turns must not pollute the original context, so it needs its
  own transcript — a new file, parent-linked, which is exactly what sub-agent
  runs already do. Reverting is reopening the parent. No tree needed, because the
  relation is between files rather than inside one.
- **A permanent retarget.** "This agent is now something else." Continuous with
  what came before, so it is a record appended in place, consistent with how
  compaction and clear were settled above.

The leaning is one command whose behaviour follows from whether a new transcript
was requested. That needs arguing rather than assuming, since one command with
two behaviours is a smell unless the two map cleanly onto two intents.

## Open question C — Remaining details

- **Is a profile bound to a session, or orthogonal?** Orthogonal is cleaner, but
  then "revert" is two operations rather than one.
- **How does switchable source configuration keep the property it has now?**
  Sources are immutable at `RpcServer.__init__` deliberately, so a reload cannot
  silently widen a registry an embedder restricted. Making them switchable must
  preserve that.
- **Collision and validation.** First-wins-and-warn matches both existing
  loaders. Unresolved: what happens when a profile names a tool or hook source
  that does not exist — refuse the profile, or load it degraded and log. Refusing
  is probably right, since a profile that silently grants fewer tools than it
  claims is the same class of lie as a partial switch.
- **File intelligibility.** A file's relationship to its origin should be
  explicit rather than inferred. Sub-agent transcripts are currently identifiable
  only incidentally, by having `parent_tool_call_id` set; that wants an `origin`
  discriminant. A forward `continued` record in the parent would also make a
  chain walkable in both directions instead of only backwards.

## Consequences

- **#49's storage redesign closes.** Its useful remainder is session discovery —
  nothing enumerates sessions today, so a name is only readable on a session you
  have already opened.
- **#57 stays a bug and gains a constraint**: its fix should anticipate a single
  profile record rather than separate `identity` and `model_change` records.
- **`switch_session` is not a primitive.** It is a courtesy wrapper over "apply
  profile, open session", and it only works once #57 makes configuration durable.
  An earlier note argued to drop it for lack of a consumer; profiles are the
  consumer.
- **Sub-agents are the existing proof of this shape.** File-per-run, parent
  pointer, own prompt, own tool subset. This ADR generalises from them rather
  than treating them as a special case — and if `SubagentSpec` and a profile
  converge, #44 becomes the first implementation of this ADR rather than a
  neighbour of it.
- **The HTML exporter is unrelated dead weight** and is being removed separately.
