# Roadmap to MVP

midge does what it set out to do: every subsystem in the README exists and is unit-tested. Running
it end to end is a different test. A containerised run against a real model, followed by a
hand-driven TUI session (branch `test/docker-harness`, `e2e/FINDINGS.md`), filed most of the
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

*Status as of 2026-10-02: M1, M2 and M3 done; the release milestone, now scoped as a deployable package, is what remains. This is a living document: update it in the PR that closes an item.*

---

## M1 — Trustworthy daily driver ✅

**Goal:** midge can be relied on as a personal coding agent in the TUI.

**Done**, in six PRs into `develop`, each merged only after a behavioral test in the container
(`e2e/`, brought in by #122) against a real model — RPC scripted under `e2e/scenarios/`,
the TUI driven through tmux. The exit criteria, as met:

- **No open issue kills a turn or corrupts a conversation.** #99 was pulled in from M2 once it was
  clear RPC is the primary mode: an unattended session that forgets everything on restart is the
  worst case, not an embedding detail.
- **Every safety claim is true as written.** The explorer is read-only because its allowlist is;
  the docs say a hook gates tool calls, not effects.
- **A fresh install fails in sentences.** A missing key is on screen before the first prompt; a
  rejected one is a sentence, not a 401 repr; a bad log path is a warning.
- **Interrupting a turn, including during compaction, has tests** — and writing them found a
  cancel during compaction was being swallowed.

| Item | Issue | PR |
|---|---|---|
| RPC never wrote the conversation to its transcript | #99 | #123 |
| One turn runner for both front-ends; RPC auto-compacts too | — | #123 |
| Interrupt during compaction; a swallowed cancel inside it | #114 | #123 |
| Interrupt-persistence tests | — | #123 |
| Bubbles parsed content as markup, killing the turn | #110 | #123, #124 |
| Ctrl+C kept the steering queue | #112 | #124 |
| Alt+Enter submitted on macOS — Ctrl+O is the newline key | #113 | #124 |
| An empty assistant turn poisoned every later request | #111 | #125 |
| A tool's own `KeyError` reported as "tool not found" | #104 | #125 |
| `edit` near misses say what to fix | #97 | #125 |
| A log file in a missing directory crashed startup | #105 | #125 |
| Tool calls in one message raced — mutating tools now run alone, in order | #101 | #126 |
| The "read-only" explorer could write — `ls` and `grep` added | #103 | #127 |
| Hook docs overclaimed; `allowlist_extension` added | #102 | #127 |
| TUI approval prompt, on by default, never in RPC | — | #127 |
| Missing or rejected API key as a sentence | — | this PR |
| `--continue`, `--version`, token counts in the header, README flags | — | this PR |

---

## M2 — Embeddable via RPC ✅

**Goal:** another program can depend on `midge --rpc` the way it depends on a library.

**Done**, in three PRs into `develop`, each behaviorally tested in the container against a real
model and, where it applied, shown failing against `develop` first. The exit criteria, as met:

- **A client learns the protocol version without asking.** The server's first frame is
  `{"type": "ready", "protocol": 1, "midge": …}`; `get_state` carries both.
- **Every command, response and event is documented in one place** — [`docs/rpc.md`](docs/rpc.md),
  with a compatibility policy — **and a test fails when one changes.** `tests/test_rpc_contract.py`
  pins 39 frame shapes in a golden file and fails if the reference stops naming a frame or command
  the source has. A real-model session in the container produced no shape outside it.
- **A subprocess test** (`tests/test_rpc_subprocess.py`) spawns `python -m midge --rpc` against a
  stdlib fake OpenAI server, runs a turn, ends the process, starts another on the same session, and
  checks the model was sent the first turn.
- **No malformed input ends the process** — an over-long line, a bad `id`, not-JSON — and, found
  while writing the contract test, **no failing operation does either**.

| Item | Issue | PR |
|---|---|---|
| An over-long line killed the server | #100 | #131 |
| A non-string `id` was dropped and the command ran | #106 | #131 |
| Exception reprs on the wire | #107 | #131 |
| Sub-agents bound twice | #109 | #131 |
| CI on `develop`; `python -m midge` | — | #131 |
| Protocol version and `ready` | — | #132 |
| `docs/rpc.md` plus golden-frame test | — | #132 |
| An operation's unexpected exception ended the process | — | #132 |
| Subprocess end-to-end test | — | this PR |
| `examples/rpc_client.py` replaces `examples/rpc_agent.py` | — | this PR |
| Stale `src/midge/rpc.py` references | — | this PR |

---

## M3 — Retargetable base ✅

**Goal:** changing domain means changing extensions, config and a profile, and nothing else.

**Done**, in three PRs into `develop`, each behaviorally tested in the container and shown failing
on `develop` where it applied. The exit criteria, as met:

- **The notes domain runs as `midge --extension-dir examples/notes_extension --profile notes`.**
  `examples/notes_agent.py` — a hand-copied wiring script that had drifted until it no longer ran —
  is gone. In the container the notes agent added and listed notes, asked approval only for the
  mutating tool, and had no shell to run.
- **A domain runs without the coding tools and still uses skills.** `[tools] builtin = false`
  loads none; skills stay reachable through `/skill:`, and the catalogue appears whenever any tool
  declares it can open a file by path.
- **`docs/retargeting.md` walks a new domain from an empty directory**, and a test builds that
  domain from the guide's own code blocks, so it cannot drift.

| Item | Issue | PR |
|---|---|---|
| The base system prompt was hardcoded — `[agent] system_prompt` | — | #135 |
| The coding tools always loaded — `[tools] builtin` | — | #135 |
| Skills gated on a tool named `read` — `@tool(reads_paths=True)` | — | #135 |
| A profile without a reader was shown the catalogue at startup | — | #135 |
| `pi_bash_` spill-file prefix | #108 | #135 |
| The notes domain through the real entrypoint; `notes_agent.py` removed | — | #136 |
| `docs/retargeting.md`, tested against its own examples | — | #136 |
| Palette vs drawer: verbs in Ctrl+P, everything switchable in the drawer | #115 | this PR |

---

## Release — a deployable package

**Goal:** midge is something another project depends on by version, not by commit. A deployer pins
a tag, installs only what an unattended agent needs into a sandbox image, type-checks its own
extensions against midge, and drives `midge --rpc` from a host process with a client that moves
in step with the protocol.

As of `main` at `e0d1ae0`, the wheel builds, installs from git into a clean 3.12 venv, and
`midge --version` runs. What is missing is everything that makes that repeatable and lean.

**Exit criteria:**

- **A tag to pin to.** `v0.1.0` exists, the version has one source, and the sandbox kit's `ref`
  can name a tag rather than a commit.
- **The installed package is tested, not just the source tree.** CI builds the wheel, installs it
  into a clean environment, and runs `midge --version` and a scripted RPC prompt against it.
- **A headless install carries no TUI.** `pip install midge` is enough for `midge --rpc`;
  `midge[tui]` adds Textual.
- **Downstream code type-checks against midge.** An extension's `@tool` functions and `Profile`
  declarations get midge's annotations, not `Unknown`.
- **A host has a supported client.** Embedding no longer means copying `examples/rpc_client.py`.

| Item | Issue | Size | Notes |
|---|---|---|---|
| One source for the version, a CHANGELOG, tag `v0.1.0` | **proposed** | S | The version lives in `pyproject.toml` and `__init__.py`, and there are no tags. Deployers, and the sandbox kit, pin a commit hash today. |
| Ship `py.typed` | — | S | Done. |
| Textual as an optional `tui` extra | — | S | Done. |
| Build the wheel and smoke-test the installed `midge` | **proposed** | S | `poetry check` passes, but nothing proves `pipx install` works. Run against the built wheel in a clean venv, including one RPC round trip, and without the `tui` extra. |
| CI across Python 3.11–3.13 and macOS | **proposed** | S | CI runs 3.11 on Ubuntu only, and development happens on macOS. |
| A packaged RPC client | **proposed** | M | The only client is `examples/rpc_client.py`, which a host copies and then maintains against a protocol that moves without it. Ship it as `midge.rpc.client`: standard library only, as the example is, typed, checking `ready`'s protocol version, and covering the commands an embedder needs (prompt, abort, `use_profile`, `clear_context`, sessions). Tested against a real `midge --rpc` subprocess. |
| Sandbox kit and Dockerfile install a release | **proposed** | S | Both build from source; once a tag exists they install the wheel at that tag, so an image is the same artifact a deployer pins. |
| `pyproject` description says "~2k LOC" | — | S | Done. |
| CLAUDE.md says `use_profile` is pending | — | S | #60 and #67 closed on 2026-07-30. Done. |
| CONTRIBUTING.md says to branch off `main` | — | S | CLAUDE.md's git flow cuts feature branches from `develop`. Done. |

---

## Decisions needed

These shape items above. None of them is settled by writing them down here.

- **A second wire format (e.g. Anthropic).** Currently under *Not MVP*: the provider registry makes
  it an adapter rather than a branch, but CLAUDE.md asks for a real reason, not symmetry.
- **Settings for extensions.** An extension has no `Config` of its own, so the notes example reads
  `MIDGE_NOTES_KB` from the environment. Worth solving once a domain needs more than a path.
- **Where the RPC client ships.** Inside `midge` is simplest and keeps client and protocol in one
  version, but a host then installs midge's dependencies to talk to a pipe. A separate,
  dependency-free distribution avoids that at the cost of a second release. With Textual optional,
  the cost of the first is openai, pydantic, pyyaml and tenacity.
- **PyPI or tags only.** A public repo with tags is enough to pin by version. Publishing to PyPI
  needs the name to be free and a release workflow; decide before `v0.1.0` so the CHANGELOG says
  how to install it.

Decided during M1, recorded so they are not reopened by accident:

- **midge is RPC-first.** The TUI is the human-facing mode; the unattended one is primary. Anything
  that needs a person in the loop lives in `tui/` and never blocks RPC.
- **#101:** anything not read-only runs alone and in order; reads run together. `@tool(read_only=True)`
  declares it; a sub-agent derives it from its allowlist.
- **Approval** is TUI-only and on by default (`[tui] approve_tools`). RPC never asks — the boundary
  there is the container.
- **#115:** Ctrl+P lists verbs; the drawer (Ctrl+B) is the one place to switch session, profile or
  model. A section with nothing to pick says why rather than vanishing.
- **The skills catalogue** is shown when a tool declares `reads_paths`, and names that tool.

## Not MVP

- Everything in CLAUDE.md's out-of-scope list: OAuth, a pi-tui port, pi-mom/pods/web-ui,
  WASM/native deps, LangChain/LiteLLM.
- Cost and price tables. Token counts shipped in M1; prices churn like model lists, and midge ships no
  model list for the same reason.
- A socket transport. midge never listens; bridging is the deployer's choice.
- Shipping the end-to-end test rig as part of midge. It is in the repo (`e2e/`, #122) as the
  merge gate for behaviour a unit test cannot see, not as something a user installs.

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
- **Docs:** `notes/` replaced by subsystem docs in `docs/` (#76).
