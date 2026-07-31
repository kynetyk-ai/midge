# Findings

Observations from driving a containerised midge over RPC. One entry per finding,
each labelled by whose fault it is — **midge**, **the model**, or **neither**
(my expectation was wrong, or the behaviour is deliberate). That distinction is
the point of the exercise; a claim gets checked against the source before it is
written down here.

No issues are filed from this branch.

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
