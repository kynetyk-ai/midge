# Findings

Observations from driving a containerised midge over RPC. One entry per finding,
each labelled by whose fault it is — **midge**, **the model**, or **neither**
(my expectation was wrong, or the behaviour is deliberate). That distinction is
the point of the exercise; a claim gets checked against the source before it is
written down here.

**Filed as issues #99–#109 on `main`.** The mapping:

| finding | issue |
|---|---|
| F17 RPC never persists messages | [#99](../../issues/99) |
| F1 over-long line kills the server | [#100](../../issues/100) |
| F7 tool calls in one message race | [#101](../../issues/101) |
| F13 + F14 hooks gate tools, not effects | [#102](../../issues/102) |
| F15 "read-only" explorer can write | [#103](../../issues/103) |
| F12 a tool's `KeyError` becomes "tool not found" | [#104](../../issues/104) |
| F2 log file in a missing directory | [#105](../../issues/105) |
| F3 non-string `id` silently dropped | [#106](../../issues/106) |
| F4 + F11 exception reprs on the wire | [#107](../../issues/107) |
| F8 `pi_bash_` spill prefix | [#108](../../issues/108) |
| F16 sub-agents bound twice | [#109](../../issues/109) |

F5, F6, F9 and F10 are not filed: two were my expectations being wrong, two were
the model behaving badly while midge behaved correctly. F10 — that this model
prefers `bash` over `edit` for mutations — is cited inside #102 as context,
because it is what makes the denylist the main path rather than an edge case.

---

## Phase 0 — the protocol, no model spend

37 command probes, 8 raw-input probes, 3 startup-refusal runs, 1 limit probe.
The dispatch loop survived every malformed input except one.

### F1 · An oversized line kills the server — **midge**

**Severity: high.** One client mistake terminates the agent and loses the
conversation.

Sending a single line larger than `READ_LIMIT` (16 MiB) does not produce a
parse error. It raises out of `readline()`, through `serve()`, through
`serve_stdio`, out of `main`, and the process exits 1:

```
File ".../rpc/transport.py", line 117, in read_line
    return await reader.readline()
File ".../asyncio/streams.py", line 571, in readline
    raise ValueError(e.args[0])
ValueError: Separator is not found, and chunk exceed the limit
```

Repro: `docker exec midge-harness python -c "…"` writing a 17 MiB line into the
FIFO. The container goes from serving to exited.

Why it is a defect rather than a limit doing its job: **every other malformed
input is answered and the loop continues.** Not-JSON, a bare array, a bare
string, `null`, a missing `type`, a non-string `type` — all get
`{"command": "parse", "success": false, …}` and the server keeps serving. An
over-long line is the one input that is fatal, and it is the one a client is
most likely to send by accident, since `prompt` is where a large paste lands.

The transport docstring names this exact failure as the reason the limit was
raised from 64 KiB:

> `READ_LIMIT` exists because the default 64 KiB would turn a large pasted
> prompt into a `ValueError`

Raising the ceiling moved the cliff without removing it. `asyncio` leaves the
oversized data in the buffer, so recovering means draining to the next newline
and answering with a parse error — the same shape every other bad input gets.

Note this is *not* the documented EOF behaviour. EOF is a deliberate "the parent
is gone, so should we be"; this is a client that said too much on one line, and
the conversation dies with it.

### F2 · A log file in a missing directory crashes at startup — **midge**

**Severity: medium.** Found by accident while bypassing the entrypoint.

`MIDGE_LOG_FILE=/nope/does/not/exist.log` produces a raw traceback and a dead
process:

```
FileNotFoundError: [Errno 2] No such file or directory: '/nope/does/not/exist.log'
```

`logs.configure` constructs `logging.FileHandler(log.file)`, and `FileHandler`
does not create parent directories.

The inconsistency is what makes it a defect. `config.py`'s stated contract is
that **bad input is a diagnostic, never an exception** — a malformed file, an
unknown key, a wrongly typed value all degrade to the default and say so. A
mistyped `[log] file` is the one config value that kills the process. And
`Session.new` already does `p.parent.mkdir(parents=True, exist_ok=True)` for
exactly this case, so persistence creates its parent while logging does not.

Worse in RPC mode specifically: this happens *before* logging exists, so the
traceback goes to bare stderr with none of the diagnostic machinery around it.

### F3 · A non-string `id` is silently dropped — **midge**

**Severity: low**, but the asymmetry is the argument.

```
sent: {"id": {"a": 1}, "type": "get_state"}
got:  {"type": "response", "command": "get_state", "success": true, "data": {…}}
```

No `id` on the response. `server.py:185` is
`cmd_id = cmd_id_raw if isinstance(cmd_id_raw, str) else None`, so a non-string
id becomes `None` and the command **still executes**.

Every other malformed field is refused: `roots_only` that is not a boolean,
`targets` that is not a list, a `transcript` outside the three modes. The id is
the one field where a wrong type is silently accepted — and it is the field a
client uses to correlate, so the failure is a client that ran a state-changing
command and can never learn whether it worked. With `new_session` that means a
timeout, a retry, and a second session.

### F4 · Validation errors put internal model names on the wire — **midge**, cosmetic

```
"error": "1 validation error for UseProfileParams\ntranscript\n  Input should be
 'continue', 'fork' or 'resume_last' …"
```

`UseProfileParams` and `ReloadParams` are implementation detail. The text is
multi-line, though JSON-escaping keeps each frame on one line so the protocol
holds. Compare the hand-written refusals — "`model` is required and must be a
non-empty string" — which say the same thing without naming a class.

### F5 · `reload` is always configured from the CLI — **neither**

I expected `reload` to refuse with "no sources configured". It succeeded.

Correct: `cli.py` builds `extension_sources = [*BUILTIN_TOOL_DIRS, …]` and
`skill_sources = [*args.skill_dir, *default_skill_dirs()]`, and neither list is
ever empty, so both targets are always "configured". The refusal is only
reachable by an embedder passing `None` — `examples/rpc_agent.py` does. My
expectation was wrong, not the code.

### F6 · A blank line is deliberately unanswered — **neither**

`"   "` gets no response at all. That is intended, and the code says so: *"a
whitespace-only line is a blank line, not a malformed command, and answering it
with a parse error is noise."*

### Confirmed working

- All 20 commands reachable without a turn behave, including `compact` on an
  empty history returning `{"summary": null}` without touching the provider
  (`cut_idx == 0` returns early).
- `abort` with nothing in flight, and `use_profile` naming an unknown profile,
  both refuse with a usable message.
- Startup refusals: `--profile nope` → *"profile 'nope' was not discovered"*;
  a model absent from a non-empty `[models]` table → *"model 'not-a-model' is
  not in the model registry; registered: gpt-5.4-mini"*.
- **A missing `OPENAI_API_KEY` is not a startup refusal** — midge starts and
  serves normally, as designed (`or "not-needed"`), and the 401 arrives on the
  first turn.
- `_stdout_writer` correctly took its blocking fallback for a regular file:
  `rpc_writer mode=blocking reason=not_a_pipe`.

---

## Phase 1 — the four built-in tools, against `gpt-5.4-mini`

16 scenarios, ~45 model turns. Every tool's happy path and most error branches
behaved. One finding is the most serious thing the harness has produced.

### F7 · Tool calls in one assistant message race each other — **midge**

**Severity: high.** Reproducible, and the error the model receives actively
misleads it.

Asked to copy a file and then edit the copy, the model emitted both calls in a
single assistant message. `agent.py:340-342` starts every call in a message as a
task and gathers them:

```python
for _, tc in pending:
    tool_tasks[tc.id] = asyncio.ensure_future(self._run_tool(tc))
executed = await asyncio.gather(*(tool_tasks[tc.id] for _, tc in pending))
```

So they ran concurrently, and the edit lost. From `midge.log`, same millisecond:

```
16:23:59,817  tool_start   bash  cp …/SKILL.md notes.md
16:23:59,817  tool_start   edit  notes.md
16:23:59,817  tool_failed  edit  error=FileNotFoundError
16:23:59,818  tool_ok      bash  ms=1
```

The `cp` finished 1 ms *after* the edit had already failed on the file the `cp`
was creating.

**Running independent calls concurrently is right** — three reads at once is the
point of parallel tool calls, and most parallel calls are independent. What
makes this a defect is not the concurrency but everything around it:

- **Nothing states the guarantee.** No docstring, comment or protocol note says
  calls in one message may run in any order, so a reader assumes the order the
  model emitted them.
- **The error is misleading.** The model is told `No such file: notes.md`. That
  is false — the file was moments from existing. It cannot tell a race from a
  genuine missing file, so it cannot recover correctly, and here it re-did the
  work from scratch.
- **A weak model emits dependent calls in one message.** This is not a
  hypothetical; `gpt-5.4-mini` did it unprompted on the second attempt of a
  four-call scenario.

Worth considering: reads are safe to parallelise, mutations are not obviously
so. Serialising the write-ish tools, or at least saying out loud that ordering
is not guaranteed, would both close it.

### F8 · The bash spill file is named `pi_bash_` — **midge**, cosmetic

`bash.py:81` is `tempfile.mkstemp(prefix="pi_bash_", suffix=".log")`, so large
output spills to `/tmp/pi_bash_ttdscny1.log` — and the model is *told* that path
and may repeat it to a user. midge deliberately shares vocabulary with `pi-mono`
for concepts, but this is a runtime artifact in a project called midge.

### F9 · After a missing file, the model edited a different one — **the model**

`edit src/toybox/nothing.py` correctly returned `FileNotFoundError`. The model
then searched, picked `src/toybox/text.py`, and edited *that* — changing
`truncate`'s ellipsis to `"y"` because the prompt had said "change 'x' to 'y'".

midge did exactly the right thing at every step. It is recorded because it is
the clearest argument the harness has produced for `examples/approval_extension`
existing at all: nothing in the loop distinguishes "the edit the user asked for"
from "an edit the model invented after being told no".

### F10 · The model routes around `edit`, preferring bash heredocs — **the model**

In the more complex scenarios `gpt-5.4-mini` consistently mutated files with
`bash` + a Python heredoc rather than the `edit` tool — the same pattern seen in
the real session that produced #97.

That relocates where safety lives. If a weak model's mutations mostly flow
through `bash`, then `edit`'s exactness buys less than expected, and the
`tool_call` hook's `bash` denylist matters more.

**#97 did not reproduce here.** On Python sources the model's `old_text` was
exact every time. The near miss reported in #97 was against wrapped markdown
with continuation indents, and the one markdown attempt failed via F7's race
before matching was ever attempted.

### Confirmed working

- `read` — sane `offset`/`limit`, `FileNotFoundError` surfaced as a tool error
  the model then reports honestly rather than inventing content.
- `write` — both required args, `mkdir -p` through nested paths.
- `edit` — exact matches, `first_changed_line` plus a unified diff.
- `bash` — `[exit code: 4]` surfaced and read back correctly, the 60 s default
  timeout firing with a clean `TimeoutError`, and the 2000-line cap spilling to
  a temp file the model is told about.
- The loop recovered from every tool error without a turn dying.

### A weakness in the fixture, not in midge

`combined.fix` asked the model to find the function the README calls "wrong on
purpose". It could not, and neither can I: toybox's tests all pass, so there is
no detectable defect. The fixture needs a bug that a test actually catches
before that scenario means anything.

---

## Phase 2 — skills, against `gpt-5.4-mini`

4 scenarios plus 3 deterministic probes, ~25 turns. **No midge defects.** The
skill mechanism did what it claims, including the part that depends on the model
cooperating.

### F11 · An unknown skill's error carries a `KeyError` repr — **midge**, cosmetic

```
sent: {"type": "prompt", "message": "/skill:nosuch do a thing"}
got:  {"success": false, "error": "\"No skill named 'nosuch'\""}
```

The message is wrapped in an extra pair of quotes because it is `str(e)` of a
`KeyError`, and `str(KeyError("x"))` is `"'x'"`. Same family as F4: an exception
type showing through where a written sentence was intended. The behaviour is
right — the command is refused and **no run is started**.

### Untested, and not reachable from the CLI

The skills catalogue is only injected when `read` is in the registry — *"the
catalogue tells the model to open a `SKILL.md`, so without a tool that can open
one it is an instruction to do the impossible."* I could not exercise the
negative case: `cli.py` always includes `BUILTIN_TOOL_DIRS`, so `read` is always
present. It is reachable two ways, both later: an embedder passing a restricted
registry, or a profile whose `tools` projection excludes `read`. Deferred to
phase 5, where profiles arrive.

### An observation about expansion, not a defect

`skill_message` **inlines the skill body** into the user message. Given that, a
model re-reading the `SKILL.md` it was just handed is redundant — and
`gpt-5.4-mini` did exactly that in two of three explicit invocations, spending a
turn to fetch text already in its context. It did *not* do so for
`commit-message`, so the behaviour is inconsistent rather than systematic.
Model-side waste; nothing for midge to fix.

### Confirmed working

- **The catalogue** is injected into the generated half of the system prompt,
  with `<name>`, `<description>` and an **absolute** `<location>` per skill, and
  the operator's base prompt kept separate from it.
- **Discovery across two `--skill-dir` flags**, each skill reporting its own
  `source_info.path`.
- **`/skill:name` expansion** puts the body in front of the model, and trailing
  text after the name arrives as instructions — asked to answer in one line
  instead of doing the work, it did.
- **An unknown `/skill:` name is refused before any run starts**, so a typo
  costs nothing.
- **Bundled references resolve.** The `commit-message` skill points at
  `references/style.md` by *relative* path; `skill_message` states
  *"References are relative to {base_dir}"*, and the model read the file at its
  absolute path and applied the style.
- **A skill chosen from the catalogue alone.** Given no `/skill:` prefix and
  only the catalogue, the model found `toybox-setting` by description, read it
  and its checklist, and then made all three edits the skill prescribes — which
  is the whole mechanism working without being told.

---

## Phase 3 — extensions and hooks, against `gpt-5.4-mini`

10 scenarios plus a deterministic reload probe, ~35 turns. The hook *mechanism*
is sound and verified. What the phase found is that per-tool policies do not
survive contact with a model that has `bash`.

### F12 · A tool's own `KeyError` is reported as "tool not found" — **midge**

**Severity: medium-high.** The real error is destroyed, and midge's own shipped
example triggers it.

```
call: read_note {"title": "Does Not Exist"}
got:  Tool 'read_note' not found
```

`read_note` was registered — the log shows `tool_registered tool=read_note`. The
message is wrong because `agent.py:471` wraps the whole invocation:

```python
result = await self.tools.invoke(tc.name, tc.arguments, call_id=tc.id)
...
except KeyError:
    return _tool_error(tc, f"Tool {tc.name!r} not found")
```

`ToolRegistry.invoke` raises `KeyError("Tool 'x' not registered")` for an unknown
name, then `return await t.invoke(...)` — so a `KeyError` raised *inside the tool
body* arrives at the same handler and is described as a registration failure.

`examples/notes_extension` raises `KeyError` in three places (`notes.py:120`,
`:150`, `:152`) for exactly the thing `KeyError` is for — a missing key. So
midge's own example extension trips midge's own misreporting.

Three consequences, all observed:
- The genuine message — *"No note titled 'Does Not Exist'"* — is **lost**. The
  model never learns why the call failed.
- The model is told a tool it can see in its schema list does not exist. Here it
  abandoned `read_note` and fell back to `search_notes`.
- The log says `tool_not_found`, pointing a developer at registration rather
  than at the tool body.

`KeyError` is the ordinary Python idiom for "not found", so any extension author
using it hits this. Distinguishing the lookup failure — a distinct exception
type, or a membership check before invoking — would separate them.

### F13 · The approval denylist is evaded on the first retry — **midge**, framing

**Severity: high as documented; low as an example.**

The hook works. Asked to delete a directory, the model tried `rm -rf` and was
blocked — `audit_tool_blocked tool=bash pattern=rm -rf`. Then, unprompted, it
tried again:

```
1. bash  rm -rf /work/tests                      → BLOCKED
2. bash  find /work/tests -mindepth 1 -delete    → allowed
   "Deleted the contents of /work/tests using a non-`rm -rf` approach."
```

`/work/tests` was emptied. The model narrated routing around the policy.

The `tool_call` mechanism is not at fault and is proven working. What is at
fault is what the documentation claims for it. `rpc/__init__.py` answers the
"anything that can send a line can run `bash`" risk with:

> Gating that is what a `tool_call` hook is for — see
> `examples/approval_extension/`, which applies to sub-agents too.

A regex denylist over command strings cannot be that gate. `find -delete`,
`python -c "shutil.rmtree(...)"`, `truncate`, `mv` to `/dev/null` and any number
of others reach the same outcome, and phase 1 (F10) established that this model
prefers `bash` for mutations in the first place. The example is a fine
demonstration of *how to write a blocking hook*; citing it as the answer to an
untrusted-peer risk oversells it. An allowlist inverts the burden; a denylist
enumerates what someone already thought of.

### F14 · A hook guarding `write`/`edit` is bypassed by a shell redirect — **midge**, same class

`repo_guard` refuses writes into `.midge/`. Asked to write there, the model did
not use `write`:

```
bash  mkdir -p .midge && printf 'hello' > .midge/scratch.txt   → allowed
```

Zero hooks fired. The guard inspects `tool_call.arguments["path"]` for `write`
and `edit`, and `bash` has no `path` argument to inspect.

This is my own extension's flaw, but it generalises F13 from the other
direction: **`tool_call` hooks gate tools, not effects.** With a model that can
reach the filesystem through `bash`, any per-tool policy is advisory unless it
also reasons about shell commands — which is the hard problem the denylist was
already losing.

### An observation on belt and braces

For two of three denylist patterns the hook was **never reached**: told to run
`rm -rf` or `git push --force`, the model refused on its own, citing the policy.
That is `approval_extension`'s `SYSTEM_PROMPT` doing the work before the hook
has to. Only `sudo` got as far as a tool call on the direct prompts.

Worth knowing when reading a clean log: zero blocks can mean the prompt
sufficed, not that the hook is inert. It is also why F13 needed a scenario that
explicitly invited a second attempt.

### Confirmed working

- **Blocking works when reached.** `sudo` and `rm -rf` were both genuinely
  stopped, with `ToolCallResult(block=True)` reaching the model as a tool error
  naming the pattern, and the model reporting it accurately.
- **A non-matching command passes through** untouched.
- **Extension tools sit alongside the built-ins** — 9 tools registered, and the
  model used `add_note` / `search_notes` / `read_note` without confusion.
- **`ValueError` from a tool surfaces intact**: the duplicate-slug and
  non-alphanumeric-title messages both reached the model, which explained them
  correctly. (Contrast F12 — only `KeyError` is swallowed.)
- **`reload` is idempotent.** Two reloads produced two `repo_guard_unloaded`
  lines, and one tool call afterwards produced exactly **one** observe line, so
  handlers did not stack. That is what `add_cleanup` exists for, and it holds.

---

## Phase 4 — sub-agents, against `gpt-5.4-mini`

~30 turns plus two purpose-built fixtures. The delegation machinery is in good
shape; the one finding is the third instance of the same theme.

### F15 · The "read-only" explorer can write — **midge**

**Severity: medium.** It is an example, but the false claim is in the text the
*parent model* reads when deciding whether delegating is safe.

`examples/subagent_extension/explore.py` says read-only three times:

| line | text |
|---|---|
| 1 | "a **read-only** explorer the main agent can delegate to" |
| 29 | child's own prompt: "You are read-only. **You have no tools that modify anything**" |
| 38 | tool description the parent sees: "**Not for edits: the explorer cannot change anything.**" |
| 45 | `tools=("read", "bash")` |

Tested with a canary file containing `ORIGINAL`:

```
subagent_start agent=explore depth=0 tools=2
tool_start tool=bash  {'command': "printf 'CHANGED' > /work/canary.txt"}
$ cat /work/canary.txt  →  CHANGED
```

The child mutated the workspace, and its own system prompt had asserted it
could not.

This is F13 and F14 a third time, and the sharpest version: **an allowlist
containing `bash` is not an allowlist.** #79's stated position is that "the
allowlist is the rein" and that a cap which overrides a declaration is the wrong
mechanism — which is right, and exactly why the example matters: it is the
reference for how to write one, and it demonstrates a constraint that does not
constrain. Dropping `bash`, or dropping the word read-only, would fix the
example; the general lesson is that `bash` in any allowlist makes the rest of it
advisory.

### F16 · Sub-agents are bound twice at startup — **midge**, low

`subagents_bound count=1 max_concurrent=2 timeout=45 max_timeout=60` appears
twice, same millisecond. `cli.py:376` calls `bind_subagents(...)`, and then
`RpcServer.__init__` (`server.py:97`) calls `self.controls.bind_subagents(...)`
again.

Harmless — the second binding replaces the first with identical arguments — but
redundant, and worth noting given #79 and #85 were both about re-binding losing
settings. One of the two is unnecessary.

### Confirmed working

- **The correlation envelope.** Nested events carry
  `{"agent": "explore", "agent_id": "call_…", "parent_id": null, "depth": 1}`;
  43 top-level frames in the same run carried none, so a client ignoring the key
  sees exactly the stream it saw before. `agent_id` is the spawning tool call's
  id — the same id the child's transcript records as `parent_tool_call_id`, one
  scheme rather than two. Only structural events are forwarded: tool executions
  and results, never the child's text deltas.
- **Recursion is denied precisely.** A purpose-built `looper` whose allowlist
  names its own spawn tool got `subagent_start … tools=1`, not 2 — the ancestor
  set removed `spawn_looper` and left `bash`. The shipped example cannot reach
  this path at all, since its allowlist never names a spawn tool.
- **The cyclic-allowlist warning fires** alongside it:
  `subagent_cycle agents=looper -> looper`, at startup, without taking the agent
  away — both halves of #79's position, verified together.
- **`max_timeout` clamps a declaration.** A `slowpoke` declaring `timeout=600`
  under `[subagents] max_timeout = 60` was killed at
  `subagent_timeout … seconds=60` after 62 s wall clock, and the parent received
  a clean `[slowpoke timed out after 60s]` tool result rather than an exception.
- **Multi-file session intelligibility (#62)**, in both directions:

  ```
  20260731-164805-8672.jsonl                      origin=None
    └─ "continued" → path=…explore-call_Oq8….jsonl, reason=subagent
  20260731-164805-8672.explore-call_Oq8….jsonl    origin=subagent
       parent_session=… parent_tool_call_id=call_Oq8…
  ```

### Untested

`[subagents] max_concurrent` — proving a semaphore of 2 needs several
simultaneous slow children and a minute of wall clock per run. Deferred rather
than guessed at.

---

## Phase 5 — profiles, against `gpt-5.4-mini`

~20 turns plus two purpose-built profiles. The profile machinery is sound, and
testing it uncovered the most serious defect the harness has found — which is
not about profiles at all.

### F17 · RPC mode never writes messages to the transcript — **midge**

**Severity: high.** Every RPC session records a header and nothing else.

`RpcServer._run_prompt` streams the agent's events to the wire and never touches
`self.session`. There is no `append_many` anywhere under `src/midge/rpc/`. The
TUI has it twice — `tui/app.py:545` on `AgentEnd` and `:548` on interrupt — and
RPC has it nowhere.

Direct proof:

```
agent.history after one turn : 2
lines on disk                : 1     (the header)
message records on disk      : 0
```

The asymmetry that makes it unmistakable: **a sub-agent's transcript is
complete, its parent's is empty.** `subagents.py:475` does
`session.append_many(child.history)`, so a child persists while the parent that
spawned it does not:

```
…explore-call_CxjQ….jsonl   messages=6     the child
…6d6b.jsonl                 messages=0     its parent
```

What this silently breaks, all of it shipped and tested at the unit level:

- **`--session PATH` in RPC mode records no conversation.** The file exists, is
  named, carries `profile` / `continued` / `session_info` records — everything
  except what was said.
- **`open_session` (#63) restores nothing.** `agent.history = list(opened.messages)`
  on an empty file empties the agent.
- **`resume_last` (#67) returns to the right file with no context.** This is how
  I found it: ADR 0001's motivating round trip — build, fork to a reviewer,
  return to the build thread — landed on the correct transcript and the agent
  had forgotten the conversation. *"What number did I ask you to remember?"* →
  *"I don't have a number from you to remember."*
- **Resuming after a restart loses everything.**

Why the unit tests miss it: `tests/test_rpc.py` drives `RpcServer` with a fake
provider and asserts on the wire frames, and `tests/test_persistence.py` drives
`Session` directly. Nothing asserts that a turn *through the server* lands on
disk. The TUI's equivalent is covered because `tui/app.py` does the appending
inline where it can be seen.

It also retroactively explains phase 0: `list_sessions` reported `messages: 0`
for every session, which I read as "nothing has happened yet".

### Confirmed working

**The catalogue gate — phase 2's open question, now answered.** A profile
projecting `read` away removes the skills catalogue from the prompt, and the
switch re-derives it:

```
before: catalogue=True   appended=868 chars  base="You are a coding assistant…"
use_profile blind (tools=["bash"])
after : catalogue=False  appended=134 chars  base="You cannot read files directly…"
```

That is `generated_prompt()`'s gate working live, not just at startup.

**Tool projection** narrows `agent.tools` to exactly the declared set, and the
profile's prompt replaces the base while the generated half is recomposed.

**All three transcript modes**, including the two degradations:

| requested | result |
|---|---|
| `continue` | stays on the thread |
| `fork` | new sibling `…adversarial-reviewer-0.jsonl`, history kept in context |
| `resume_last`, nothing to resume | degrades to `fork` per `[profiles] resume_fallback`, and reports both `requested` and `transcript` |
| `resume_last`, thread exists | returns to the correct file |

**`profile_hook_undecided`** fires and the profile is dropped: `no-hooks`
declares `hooks={}` while `approve` is registered, so it is registered then
removed, absent from `get_profiles`, with a named warning. Taking the profile
away rather than guessing a default is the documented position, and it holds.

**Both `approve`-dependency directions** behave as designed — the shipped
`adversarial-reviewer` loads and validates only when `approval_extension` is
loaded beside it.

### Note on severity ordering

F17 outranks everything else here. A profile switch that lands on the right
transcript is worth little when the transcript has nothing in it.
