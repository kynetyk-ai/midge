# ADR 0001 — Session profiles

- **Status:** Accepted and **implemented** — **amended 2026-07-30** (Decision 4:
  a third transcript option, `resume_last`; Decision 9: hooks are decided
  exhaustively). See [Amendments](#amendments). Every decision here now has code
  behind it: #61 (profiles), #60 (hook scoping), #62 (transcript links), #63
  (reopen), #67 (`use_profile`).
- **Date:** 2026-07-30
- **Supersedes:** the forking half of #49
- **Related:** #44 (sub-agents), #54 (reload), #55 (session markers), #57 (durability bug)

## Target functionality

Retarget a running agent — new instructions, a different toolset, a different
model — and later retarget it to something else again. The motivating case is an
agent that has just built a feature and should now review that work
adversarially, then go back to building.

A **profile** is the unit of retargeting:

| dimension | what it sets |
|---|---|
| system prompt | the durable base half; midge still appends the generated half |
| model | the provider model id used for subsequent turns |
| tools | which of the discovered tools this agent can see |
| hooks | a decision, on or off, for **every** discovered hook source |

Not part of a profile: history, and the skills catalogue.

## How this differs from `pi`

`pi` addresses two adjacent problems, and neither is this one.

**Its session tree is conversation topology.** Every entry carries an `id` and a
`parentId`, a leaf record names the current tip, and loading walks leaf-to-root
(`packages/agent/src/harness/session/session.ts`). That exists to support forking
at an arbitrary past entry and switching branches inside one file. midge's problem
is not "which past state do I return to" but "what is this agent configured as".

**Its configuration changes are independent and unnamed.** `pi` records
`model_change`, `active_tools_change` and `thinking_level_change` as separate
entry types. midge's position is that these dimensions travel together and are
worth naming: "the adversarial reviewer" is a thing, whereas a model change plus a
tool change plus a prompt change is three facts a reader must correlate and infer
were one act.

So midge takes `pi`'s append-only discipline and rejects its entry tree, while
requiring something `pi` does not have: a named, atomically-applied configuration.

## Decisions

### 1. No entry tree

No entry ids, no `parentId` on messages, no leaf pointer, no `get_tree`.
`cut_index` remains a positional offset.

This is a rejection rather than a deferral because nothing in midge forks at an
arbitrary past point. "Carry the parent's context" and "name a position in the
parent" are separable:

| what a new context starts from | needs a cut point? |
|---|---|
| nothing | no |
| a summary generated at the time | no |
| the parent's current context, verbatim | no — "here", not a position |
| the first N messages of the parent | **yes** |

Only the last requires positional identity. Its use case is landing on an exact
earlier turn, which steering serves instead.

**Cost:** an exact earlier conversation state cannot be restored.

### 2. Multi-file sessions must be intelligible

Rejecting the entry tree does not reject the relation between transcripts. A
session already spans multiple JSONL files — every sub-agent run writes its own —
and profiles may add more. Those files must state their relationship rather than
have it inferred.

Required:

- **An `origin` discriminant on `SessionHeader`** — `subagent`, `profile`,
  `fork`, or absent for a root session. Sub-agent transcripts are currently
  identifiable only incidentally, by having `parent_tool_call_id` set.
- **`parent_session` on every non-root file**, which exists today.
- **A forward `continued` record in the parent**, so a chain is walkable in both
  directions. `parent_session` alone is a back-pointer, which makes "find the
  current head of this session" a directory scan.

This is a first-class requirement, not a detail. A transcript that cannot say what
it belongs to is not an audit trail.

> **Amended 2026-07-30.** This decision is now load-bearing for behaviour rather
> than only for legibility: `resume_last` (Decision 4) resolves its candidate
> transcripts by walking this chain, and excludes sub-agent runs by their
> `origin`. Both fields moved from "an audit trail should say this" to "a switch
> cannot be implemented without this".

### 3. Existing mechanisms and gaps

| dimension | mechanism | status |
|---|---|---|
| system prompt | `set_system_prompt` sets the base and recomposes, preserving the skills catalogue and extension contributions | works in-process; not durable across resume (#57) |
| model | `set_model` | works in-process; not durable (#57) |
| tools | `_child_registry` (`subagents.py`) projects a subset `ToolRegistry` from an allowlist of names | exists |
| hooks | `_Registration.source` stamped by `_SourceScopedHooks` (`extensions.py`) | stamped, never read — no selective activation |
| transcript linkage | `parent_session` / `parent_tool_call_id` on `SessionHeader` | exists; needs `origin` per Decision 2 |
| source config | `reload` re-runs the same sources | cannot change which sources |

**Tools are filtered by projection.** `load_extensions` discovers every `Tool` in
every source directory with no filtering, and that stays. Filtering happens one
layer up: `_child_registry` keeps only names in an allowlist. A profile uses the
same shape — discover once, project a subset.

Filtering hides rather than denies at call time. A `tool_call` hook could refuse
by name, but `notes/subagents.md` is explicit that telling a model it has a tool
the registry will reject is misinformation rather than access control. A profile
changes what the agent is, so the schemas it receives must match.

Projection has a consequence worth stating: **a profile switch never changes which
sources are loaded.** Discovery stays exactly as it is, and a profile only narrows
what is projected from what was already discovered. So the property that makes
sources immutable at `RpcServer.__init__` — a reload cannot silently widen a
registry an embedder deliberately restricted — survives untouched, and profiles
need no mutable source configuration. A profile naming a tool that no source
provides simply fails validation (Decision 9).

**Hooks are the only gap.** `Hooks.clear()` is all-or-nothing, `on()` returns an
unsubscribe closure that `load_extensions` discards, and the `source` on every
registration is read only to name a file in a warning. Source-scoped activation is
required. It was designed during #54 and dropped as premature because
`load_extensions` was the only registrar; profiles are its consumer.

### 4. Switching selects a named profile, atomically, with exactly one option

> **Amended 2026-07-30.** The transcript option gained a third value,
> `resume_last`. The original two are unchanged.

A switch applies every dimension or none, and is refused while a turn is in flight
(as `reload` already is — swapping a registry mid-turn breaks tool-call/result
pairing).

**There is no revert.** A switch always names the profile to apply. Going back to
what you were doing is switching to that profile by name — there is no previous-
profile stack and no undo. That keeps the operation stateless and means a client
never has to reason about how deep it is in a sequence of excursions.

`resume_last` does not weaken this. It changes *which transcript you land on*,
never *how you name the destination*: the caller still names a profile, and the
transcript is then derived from what is on disk. There is still no stack, nothing
to pop, and no notion of "back one" — resuming the general agent's thread and
then resuming it again is idempotent, where a real undo would not be.

**A switch does not touch history.** No summarize-first, no carry-N, no
clear-on-switch. A caller who wants a clean slate runs `clear_context` after
switching; one who wants a summary runs `compact`. Composing existing commands
keeps this single-purpose; the alternative is a switch that accumulates
context-handling flags.

**The one option is the transcript**, and it has three values:

| | what it does |
|---|---|
| **continue** | stay on the same transcript; append a profile record |
| **fork** | open a new transcript, linked per Decision 2 |
| **resume_last** | reopen this session's most recent transcript under the named profile; fall back per config if there is none |

Nothing else. The distinction is worth an option because these are genuinely
different intents — an excursion whose turns should not sit in the original file,
a retarget that continues the same thread, and a return to a thread already under
way — and because a forked transcript is what makes the excursion legible later.

**What `resume_last` is for.** The motivating round trip is not one switch but
three:

1. The general agent builds a feature on the root transcript.
2. `use_profile("reviewer", fork)` — a *fork*, deliberately, so the build
   conversation cannot cloud the review.
3. `use_profile("general", resume_last)` — back to the build thread, picking up
   where it left off.

Step 3 is unserviceable by the original two options. `continue` would drag the
review into the build thread, and `fork` would open a third transcript with none
of the build context — the switch back would cost exactly the conversation the
excursion was designed to protect. Without `resume_last`, an excursion is
one-way.

**"Last" is scoped to the current session, and nothing wider.** The candidate set
is the transcripts of *this* session — the chain reachable through
`parent_session` and the forward `continued` records of Decision 2. It is not a
directory scan, not a configured sessions directory, and not a search of other
sessions. Resuming an unrelated conversation because it happened to run under the
same profile is not what the caller asked for; "where I left off" means in this
session.

Two things follow, both worth having:

- **Decision 2 becomes load-bearing rather than hygienic.** The forward
  `continued` record is what makes the chain walkable without a directory scan,
  and the `origin` discriminant is what excludes sub-agent transcripts — which
  are `origin: subagent`, not profile excursions, and must never be resumed as
  one.
- **Session discovery (#64) is not a dependency.** `resume_last` needs no listing
  and no default session location.

> **Footnote, #62.** midge has since grown a default session location —
> `.midge/sessions`, configured by `[session] dir`, with every run recording a
> transcript unless told not to. That does not disturb anything above. The
> load-bearing reason for rejecting a directory-scan `resume_last` is semantic —
> "pick up where I left off" means *in this session* — and it is untouched:
> `resume_last` still walks the chain, still resolves nothing by listing a
> directory, and still does not depend on #64. What changed is only that
> "midge would have to impose a session location" is no longer available as a
> supporting cost, since it now has one for unrelated reasons.

**Which profile a transcript is "under" is the last profile record in it**, per
Decision 5. A thread that has itself since switched away is not a candidate, and
this needs no new state — it is the existing rule read backwards.

**With no prior transcript, the fallback is configured, not invented.** The first
use of a profile in a session has nothing to resume, and that is an ordinary
first run rather than an error: the switch must still succeed, because a client
cannot know whether a profile has been used before without asking. The fallback
is `fork` or `continue`, named in config (`[profiles] resume_fallback`), and the
response says which happened so a client can render it. **The shipped default is
`fork`** — a caller reaching for `resume_last` is working in a thread-per-profile
model, and `fork` is that model's first-run behaviour, where `continue` would
silently merge two threads that the option exists to keep apart.

Atomicity is why this is one operation rather than documented guidance. A client
hand-orchestrating a switch today gets `success: true` from `set_system_prompt`
while the entire previous toolset and every hook stay active — extension and skill
sources are fixed at `RpcServer.__init__` and `reload` only re-runs the same ones.
No client discipline fixes that, because the capability to change sources is
absent.

### 5. A profile is a property of a session

A session records the profile it is running under. Switching appends a record;
the session's current profile is the last one recorded, consistent with how
`session_info` works (#55).

This makes resume correct by construction: reopening a session restores its
profile, which is what makes `switch_session` a courtesy wrapper rather than a
primitive. It depends on #57, since a profile that silently reverts on resume is
not something to build on.

### 6. Mutable state stays append-only

Anything mutable is an appended record replayed at load, last write wins; never a
header rewrite. Two independent reasons that point the same way:

1. **Traceability.** midge is often driven over RPC, where the transcript is the
   only durable account of what happened.
2. **Crash recovery.** `read_transcript` salvages a truncated final line
   (`persistence.py:146`) only because nothing earlier is ever rewritten.

A profile switch is therefore recorded, so any message is attributable to the
configuration that produced it — including the model, which matters for cost and
behaviour attribution.

### 7. Compaction and clear stay in-thread

An earlier proposal had each seal the file and start a new one, so that one file
would be exactly one context window. Rejected: the motivation was traceability,
and both operations are already non-destructive. `append_compaction` and
`append_clear` write one record and collapse only the in-memory view; the file
keeps every message and `cut_index` says which were folded. What is lossy is the
agent's forward knowledge, which is a property of compaction as a technique rather
than of the transcript.

Forking on compaction would buy only that an export shows precisely what the agent
saw. That is elegance, not audit, and it would make the TUI's automatic threshold
silently spawn files and stop `--session PATH` naming the file being written.

### 8. A profile is a declared, auto-discovered Python file

A profile is configuration, not something composed per call, so it is declared
once and discovered — the same treatment tools and sub-agents get.

The format is a **`.py` file containing an instance of a predefined dataclass**,
collected from the module namespace exactly as `load_extensions` collects `Tool`
instances today:

```python
from midge.profiles import Profile

ADVERSARIAL = Profile(
    name="adversarial-reviewer",
    description="Reviews recent work looking for what is wrong with it.",
    model="gpt-4o",
    tools=("read", "bash"),
    hooks={"approve": True, "audit": False},
    prompt="""
        You are reviewing work that has just been done. Assume it is wrong and
        find out how. Cite `path:line` for every claim.
    """,
)
```

Chosen over a Markdown file with frontmatter because a profile intersects code:
its `tools` and `hooks` fields name symbols that must exist, and the fields
themselves have a shape worth enforcing. A dataclass gets structural validation at
import and is checkable by `pyright`; frontmatter defers every error to runtime.
That a system prompt is prose is a real argument for Markdown, but a triple-quoted
string is adequate and does not justify a second file format.

Since profiles are `.py` files holding instances, `load_extensions` can collect
them with no new loader and no new flag: one file may declare a tool, a sub-agent
and a profile together. A dedicated `--profile-dir` is the alternative.

**`Profile` does not converge with `SubagentSpec`.** Their fields nearly coincide —
`name`, `prompt`, `tools`, `model` — but they are different concepts: a sub-agent
is *a tool the agent uses*, a profile is *what the agent is*. One is invoked by the
model as a delegation; the other is applied by an operator as a reconfiguration.
Unifying them would put a `timeout` on a profile and imply that either can stand in
for the other. The overlapping fields are acceptable duplication.

### 9. A bad profile is skipped with a warning

> **Amended 2026-07-30.** `hooks` must decide every discovered source, and a
> profile that leaves one out does not load. `tools` is unchanged.

Matching how extensions and skills already handle malformed input: log a warning,
skip the file, carry on. A profile naming a tool or hook that does not exist does
not load, so asking to switch to it fails at the switch with a clear error rather
than silently granting fewer tools than it claims. Name collisions are
first-wins-and-warn, as `tool_name_shadowed` and `skill_name_shadowed` already do.

**`hooks` is a decision per source, not a list of active ones**, and a source
left undecided is `profile_hook_undecided` — a validation failure like any
other. `tools` stays an allowlist.

The asymmetry is the reason. A profile that omits a tool yields a *less* capable
agent; a profile that omits a hook would yield an *unguarded* one. A profile
reads like a convenience layer, so the natural authoring mistake is to write only
what you mean to change — and there is no safe default for that mistake:

| reading | omission means | outcome |
|---|---|---|
| "the list is what is active" | unnamed hooks are off | an approval gate silently disappears |
| "the list is what changes" | unnamed hooks are on | a profile can never turn anything off |

Both are silent answers to a question the author did not know they were being
asked. Requiring a decision per source makes the question unaskable rather than
choosing a better default for it, and converts the mistake into a startup
diagnostic that names the source you forgot.

**Cost, stated plainly:** installing a new hook-bearing extension invalidates
every existing profile until each names it. That is the trade — a loud failure
that takes one line to fix, over a silent capability grant.

### 10. A central config names the default profile

> **Amended 2026-07-30.** A second profile key follows from Decision 4:
> `[profiles] resume_fallback`, `fork` (the default) or `continue`, deciding what
> `resume_last` does when the named profile has no prior transcript in this
> session. It belongs here for the reason this decision already gives — it is a
> setting, and settings live in one discoverable file.

Discovery supplies the set of profiles; something has to name the one to start
under. That is a central config file rather than another flag.

midge has no config file today — everything is command-line flags plus eight
environment variables. Introducing one is a real decision, and the reason to take
it here is that a config file is **self-documenting** in a way scattered
environment variables are not: a reader sees what is configurable in one place,
with the defaults visible.

## Follow-on work

- **Consolidate ad-hoc configuration into the config file**, as a separate
  workstream. Today: `MIDGE_MODEL`, `MIDGE_INCLUDE_USAGE`, `MIDGE_LOG_LEVEL`,
  `MIDGE_LOG_LEVEL_OPENAI`, `MIDGE_LOG_FILE`, `MIDGE_LOG_PAYLOAD_CHARS`,
  `OPENAI_BASE_URL`, and `OPENAI_API_KEY`. All but the API key are candidates —
  a credential belongs in the environment, not in a file that gets committed.
  Worth doing on its own rather than smuggled in alongside profiles.

## Consequences

- **#49's storage redesign closes.** Its remaining useful work is session
  discovery: nothing enumerates sessions, so a name is only readable on a session
  already open.
- **#57 gains a constraint.** Its fix should record one profile entry rather than
  separate `identity` and `model_change` entries.
- **`switch_session` is not a primitive** — it is "apply profile, open session",
  and it requires #57.

> **Footnote, #63.** `open_session` exists, and reopening restores history and
> the base prompt but **not** the model. The two halves of "identity" are not
> the same kind of thing: a prompt is part of what the conversation *is*, while
> a model is infrastructure with its own config key. So a recorded model is a
> *stored prior choice* that takes part in precedence — it beats a default and
> loses to one the operator asked for this run — rather than overriding
> everything the way #57 left it. Mid-run the running model always wins, since
> a `set_model` a minute ago is unambiguously a live choice. Disagreements warn
> rather than block, which also means a session recorded against a since-retired
> model no longer refuses to start.
>
> **Superseded in part by #67**, as anticipated. A switch now records a profile,
> and both resume paths restore it: the recorded profile wins over `[profiles]
> default` and loses to `--profile`, which is the same precedence the model
> follows. The base prompt is its fallback, and a recorded profile that is no
> longer discovered warns (`resume_profile_unavailable`) rather than refusing.
> What stands unchanged is the model rule — a profile names its own model, so a
> restored profile sets it, and a *loose* recorded model is still only a stored
> prior choice.
- **Sub-agents are the existing proof of this shape**: file-per-run, parent
  pointer, own prompt, own tool subset. They stay a separate concept (Decision 8).
- **`Hooks` grows source-scoped activation**, the one new mechanism.
- **midge gains a config file**, which it has not had. Everything configurable has
  been a flag or an environment variable until now.
- **An excursion is a round trip, not a one-way door** (amendment). `resume_last`
  makes fork-and-return the expected shape, which promotes Decision 2 from
  legibility to mechanism and makes #62 and #63 hard dependencies of the switch.
  Session discovery (#64) is not one.

## Amendments

### 2026-07-30 — hooks are decided exhaustively

**What changed.** Decision 9 now requires `hooks` to state a decision for every
discovered hook source; the field is a mapping rather than a list of active
names. Decision 3's description of the hooks dimension is reworded to match.
`tools` is untouched.

**Why.** The original wording — *"which registered hook handlers are active"* —
did not say what an unmentioned source does, and both available answers are
unsafe in different directions (see the table in Decision 9). The error analysis
that settled it: a profile is a convenience layer, so the instinctive way to
write one is to list only what you are changing, and under "the list is what is
active" that instinct silently removes an approval gate. Failing toward *less*
capability is tolerable; failing toward *more* is not.

**What was considered and rejected.** A second class of hook — "global",
registered by an extension declaring `ALWAYS_ACTIVE` and immune to profiles.
Rejected as a second vocabulary that treats the symptom: it makes the dangerous
omission survivable without making it visible, and it leaves the author still
guessing whether their list is a delta or a complete set. Also rejected: reading
immunity off the *absence* of a source name, which conflates "this is deliberate
policy" with "this handler happened not to come from an extension file".

**Consequence for the mechanism.** `Hooks.set_active_sources` still ignores
unnamed registrations, but that is now understood as mechanical rather than a
policy: source names are the vocabulary, so an unnamed handler could never be
switched back on. Protecting a gate from being switched off is validation's job,
not the registry's.

### 2026-07-30 — `resume_last`, a third transcript option

**What changed.** Decision 4's transcript option went from two values to three.
Decision 2 was re-scoped from legibility to mechanism, and Decision 10 gained
`[profiles] resume_fallback`. Nothing was reversed: `continue` and `fork` behave
exactly as originally accepted.

**Why.** The original two options made an excursion one-way. The motivating flow —
build on the general agent, fork to a reviewer so the build conversation cannot
cloud the review, then return to the build thread — has no third step under the
original design, because `continue` would drag the review back into the build
thread and `fork` would abandon the build context entirely. The option that
protects the review is the same option that strands it.

**What was considered and rejected.** Searching for the last conversation under a
profile *outside* the current session — a directory scan of siblings, or a
configured sessions directory. Rejected: "pick up where I left off" means in this
session, and resuming an unrelated conversation because it shared a profile is
not what the caller asked for. It would also have forced midge to impose a
default session location, which it has consistently declined to do, and made
`resume_last` depend on session discovery (#64). Scoping to the session's own
transcript chain avoids all three.

**What it does not reopen.** Decision 1 (no entry tree) stands: `resume_last`
names a whole transcript, not a position inside one, which is precisely the case
Decision 1 said needs no cut point. Decision 4's "no revert" stands too — the
caller still names a profile, never a stack position, and resuming twice is
idempotent where an undo would not be.
