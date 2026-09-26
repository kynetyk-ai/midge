# Roadmap to MVP

midge does what it set out to do: every subsystem in the README exists and is unit-tested. Running
it end to end is a different test. A containerised run against a real model, followed by a
hand-driven TUI session (branch `test/docker-harness`, `harness/FINDINGS.md`), filed most of the
issues open today. Some break a shipped feature without saying so. Two make a safety claim that
does not hold.

So "MVP" here does not mean more features. It means **what midge already claims, made true**, in
three stages:

1. **Trustworthy daily driver.** You can use midge as your own coding agent in the TUI and rely on
   it. No known bug kills or silently corrupts a turn, and every safety claim is accurate.
2. **Embeddable via RPC.** Another program can spawn `midge --rpc`, drive it, restart it, and get
   its conversation back, with a protocol it can pin to.
3. **Retargetable base.** A non-coding domain runs through the real `midge` entrypoint with config,
   extensions and a profile, and no core edits.

The order matters. RPC is the TUI's control surface with a different transport, so a loop you
cannot trust interactively is no better embedded. Retargeting is the point of the design, but it
promises the same core under a different prompt, so the core has to hold first. A **Release**
milestone closes the MVP.

**Reading this file.** An `#NN` is an open issue unless it appears under *Already done*. Items
marked **proposed** have no issue yet and are filed when work on them starts. Sizes are rough: **S**
is an afternoon, **M** a few sessions, **L** needs a design pass first. Within a milestone, items
are listed roughly in the order worth doing them.

*Status as of 2026-09-26. This is a living document: update it in the PR that closes an item.*

---

## M1 — Trustworthy daily driver

**Goal:** midge can be relied on as a personal coding agent in the TUI.

**Exit criteria**

- No open issue kills a turn or corrupts a conversation.
- Every safety claim in the docs, examples and tool descriptions is true as written.
- A fresh install gets from nothing to a first answer, and each way that can fail (no key, bad key,
  bad config) produces a sentence the user can act on.
- Interrupting a turn, including during compaction, has a test.

### Turns that die, or conversations that break

| Item | Issue | Size | Notes |
|---|---|---|---|
| Bubbles parse content as Textual markup | #110 | S | `markup=False` on the three bubble widgets, as `StatusLine` already has. Also truncate tool arguments the way results already are. |
| An empty assistant turn poisons every later request | #111 | S | `repair_history` drops content-less assistant messages. The bad message is also persisted, so resume inherits it. |
| Interrupt during compaction is silent and can un-compact | #114 | S | Widen the `try`. Keep the write order. |
| Ctrl+C leaves the steering queue intact | #112 | S | Route through `Controls.abort()`. Whether the model is told it was interrupted is a separate question. |
| Alt+Enter submits on macOS terminals | #113 | S | Add a binding that macOS terminals actually send. Stop advertising one they don't. |
| Tests for interrupt persistence | **proposed** | S | None exist. #110 and #114 are both holes in this path. |

### Errors a model or a person can act on

| Item | Issue | Size | Notes |
|---|---|---|---|
| A tool's own `KeyError` reported as "tool not found" | #104 | S | A dedicated `ToolNotFound`. The shipped `notes_extension` trips it. |
| `edit` near-miss gives nothing to recover from | #97 | S–M | Hints from `difflib`, never a fuzzy apply. Still unreproduced on markdown, where it was first seen. |
| Log file in a missing directory crashes startup | #105 | S | A diagnostic, never an exception, per `config.py`'s contract. |
| Missing or rejected API key | **proposed** | S | Today a missing key becomes `"not-needed"` and the first prompt gets the vendor's 401. Warn at startup when the provider needs a key and none is set. Render a 401 as a sentence. |

### Safety claims that are true

| Item | Issue | Size | Notes |
|---|---|---|---|
| The "read-only" explorer can write | #103 | S | Drop `bash`, or drop the claim, in all three places it is made. |
| Hooks gate tools, not effects | #102 | S–M | Reword `rpc/__init__.py`: a denylist over `bash` strings is advisory, not a boundary. Add an allowlist example. |
| Interactive approval for `bash`/`write`/`edit` | **proposed** | M | See *Decisions needed*. It would live in `tui/` plus a hook, outside the core budget. |

### Tool semantics

| Item | Issue | Size | Notes |
|---|---|---|---|
| Tool calls in one message run concurrently, with no stated order | #101 | S–M | Needs a decision first (see below), then either docs or a loop change. |

### Daily-use polish

| Item | Issue | Size | Notes |
|---|---|---|---|
| `midge --continue` resumes the most recent session | **proposed** | S | Resume exists (`--session PATH`, the switch-to panel) but needs a path. |
| Token usage in the status bar | **proposed** | S | Usage is captured (#34) and logged, never shown. Counts only; see *Not MVP* for pricing. |
| `midge --version` | **proposed** | S | `midge.__version__` exists, but no flag reads it. |
| README flag list matches `cli.py` | **proposed** | S | Missing `--skill-dir`, `--profile`, `--rpc` and `--no-session`. |

---

## M2 — Embeddable via RPC

**Goal:** another program can depend on `midge --rpc` the way it depends on a library.

**Exit criteria**

- A client can learn which protocol version it is talking to.
- Every command, response and event frame is documented in one place, and a test fails when one
  changes.
- A subprocess test spawns `midge --rpc`, runs a turn, kills the process, starts another, reopens
  the session and finds the history intact.
- No malformed input ends the process.

| Item | Issue | Size | Notes |
|---|---|---|---|
| RPC never writes messages to the transcript | #99 | S–M | **Do this first. It may be worth pulling into M1**, because it silently hollows out `--session`, `open_session` and `resume_last`. Persist in `Controls` (`commands.py`) so the TUI and RPC write turns the same way, instead of patching RPC to match. |
| An over-long line kills the server | #100 | S–M | Refuse it with a frame, drain to the newline, keep serving. |
| A non-string `id` is dropped and the command runs | #106 | S | Refuse the command. |
| Exception reprs on the wire | #107 | S | Summarise pydantic errors. Rephrase at the raise site. |
| Sub-agents bound twice at startup | #109 | S | Remove the `cli.py` call. |
| Protocol version and handshake | **proposed** | S | Nothing under `rpc/` carries a version today. |
| `docs/rpc.md` reference plus golden-frame test | **proposed** | M | Events and response shapes currently exist only as code and a module docstring. |
| Subprocess end-to-end test | **proposed** | M | All 120-odd RPC tests run in-process. This is the test that would have caught #99. |
| A minimal Python client example | **proposed** | S | `examples/rpc_agent.py` is a server launcher, not a client. |
| Stale `src/midge/rpc.py` references | **proposed** | S | In `examples/rpc_agent.py` and `notes/rpc.md`; it has been a package for a while. |

---

## M3 — Retargetable base

**Goal:** changing domain means changing extensions, config and a profile, and nothing else.

**Exit criteria**

- The notes domain runs as `midge --extension-dir examples/notes_extension --profile notes`, not
  as a separate wiring script.
- A domain can run without the coding tools and still use skills.
- `docs/retargeting.md` walks a new domain from empty directory to running agent.

| Item | Issue | Size | Notes |
|---|---|---|---|
| The base system prompt is hardcoded | **proposed** | S | `BASE_SYSTEM_PROMPT` in `cli.py:45` says "coding assistant". Make it a `Config` field. A profile can already replace it, but only once a profile is chosen. |
| Built-in coding tools always load | **proposed** | S | `cli.py:223` prepends `BUILTIN_TOOL_DIRS` unconditionally. Add a config switch. |
| Skills gated on a tool literally named `read` | **proposed** | S–M | `cli.py:231` and `commands.py:272`. A domain without `read` silently loses its skills. Gate on something a domain can declare. |
| `notes_agent.py` through the real entrypoint | **proposed** | S | Depends on the three items above. Today it re-implements the wiring and gets no profiles, RPC or skills. |
| `docs/retargeting.md` | **proposed** | M | The material is in the README. It needs a walkthrough, written last, against the result. |
| `pi_bash_` spill-file prefix | #108 | S | The model sees this name, so it leaks the old identity. |
| Palette vs drawer | #115 | — | A design question for the TUI, not a blocker. |

---

## Release — closing the MVP

| Item | Issue | Size | Notes |
|---|---|---|---|
| CI across Python 3.11–3.13 and macOS | **proposed** | S | CI runs 3.11 on Ubuntu only, and development happens on macOS. |
| Build the wheel and smoke-test the installed `midge` | **proposed** | S | `poetry check` passes, but nothing proves `pipx install` works. |
| One source for the version, a CHANGELOG, tag `v0.1.0` | **proposed** | S | The version lives in `pyproject.toml` and `__init__.py`, and there are no tags. |
| `pyproject` description says "~2k LOC" | **proposed** | S | The core is about 3.8k. |
| `notes/` has drifted from the code | #76 | S–L | Decide what `notes/` is first. |
| CLAUDE.md says `use_profile` is pending | **proposed** | S | #60 and #67 closed on 2026-07-30. |

---

## Decisions needed

These shape items above. None of them is settled by writing them down here.

- **#101: what does a message with several tool calls promise?** The options are to document that
  they run concurrently, run mutating tools one at a time while reads stay parallel, or run
  everything one at a time. The middle option is the useful one, but it means the loop has to know
  which tools mutate, which is new information for `@tool` to carry.
- **Approval: on by default in the TUI?** Today a TUI session runs `bash` with no prompt, and
  `examples/approval_extension` is a denylist rather than a confirmation. A y/n prompt on by default
  is the conventional choice for a daily driver. The cost is one more thing a hacker has to turn off.
- **#115: one surface for switching, or two?**
- **A second wire format (e.g. Anthropic).** Currently under *Not MVP*: the provider registry makes
  it an adapter rather than a branch, but CLAUDE.md asks for a real reason, not symmetry. It moves
  into M1 if you decide daily-driver means your preferred model.

## Not MVP

- Everything in CLAUDE.md's out-of-scope list: OAuth, a pi-tui port, pi-mom/pods/web-ui,
  WASM/native deps, LangChain/LiteLLM.
- Cost and price tables. Token counts are M1; prices churn like model lists, and midge ships no
  model list for the same reason.
- A socket transport. midge never listens; bridging is the deployer's choice.
- Merging the Docker test harness. It stays on its branch by design. It is how the findings were
  found, not part of midge.

## Already done

Closed recently. Listed so they are not proposed again.

- **Profiles:** discovery (#61), source-scoped hook activation (#60), `use_profile` (#67),
  default profile from config (#65).
- **Sessions:** naming and forking (#49), reopen (#63), `list_sessions` (#64), two-way links for
  multi-file sessions (#62), config surviving resume (#57).
- **Providers:** model registry (#71), token usage capture (#34), rate-limit awareness (#69),
  proactive throttling (#88).
- **RPC:** state commands and steering (#30), `get_commands` (#31), `reload` (#46), backpressure
  (#50), sub-agent correlation ids (#51).
- **Extension mechanisms:** Agent Skills (#29), sub-agents as `spawn_*` tools (#44).
- **Robustness:** interrupt during tool execution (#27), history invariants (#33), central logging
  (#35).
