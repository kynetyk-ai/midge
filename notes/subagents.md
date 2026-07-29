# Sub-agents — design notes

Implemented in `src/midge/subagents.py`. Not borrowed from `pi`: it has no
built-in delegation either, so the shape here is midge's own. The reference
material is `pi`'s *example extension*
(`packages/coding-agent/examples/extensions/subagent/`), which spawns a `pi` CLI
subprocess per sub-agent and parses its JSON event stream.

## Delegation is not supervision

These get conflated, and issue #32 was the second one:

|  | supervision (#32, declined) | delegation (this) |
|---|---|---|
| who starts the agent | an external orchestrator | the model, by calling a tool |
| where it runs | a separate process | in-process, same event loop |
| transport | Unix socket | a function call |
| what it buys | driving N workspaces from outside | keeping work out of the parent's context |

#32 was closed as not-planned: no consumer, and it adds no protocol semantics
so skipping it costs nothing if the decision is ever reversed.

## A sub-agent is a tool

That single decision does most of the work:

- **No system-prompt catalogue.** Skills need `<available_skills>` because a
  skill is not a tool. An agent-as-tool is already in the schemas the model
  receives, so this is *less* machinery than `skills.py`, not more.
- **No unknown-agent path.** A single `subagent(name, task)` tool needs a
  lookup, a failure string, and a list of valid names. `spawn_explore` needs
  none of that — the registry makes an invalid name unrepresentable.
- **Policy has a name to grab.** The `tool_call` hook sees `spawn_review` and
  can block it exactly as `approval_extension` blocks a destructive `bash`.
- **Depth control is tool subsetting.** The child's registry is a subset of the
  parent's, so "explorers may not spawn anything" is an allowlist that omits
  `spawn_*`, and the depth cap is "omit them below the limit". No counter, no
  contextvar, nothing to keep in sync. `pi`'s extension has no depth limit at
  all — with process spawning there was no natural place to put one.

The cost is that every agent's schema rides in every request. A few hundred
tokens for a handful of agents, comparable to the skills catalogue, and it
self-limits because nobody defines fifty.

## The `.py` file declares the input contract

The decorated function's **signature is the tool schema** and its **return
value is the child's opening message**. So the model fills in declared,
validated fields and cannot hand a sub-agent an arbitrary system prompt, tool
list, or model. Compare `pi`'s extension, where the model passes a free-form
`task` string and picks the agent by name at runtime.

`_build_params_model` is reused verbatim from `tools/__init__.py`, so the
parameter contract, `extra="forbid"`, and `Annotated[..., Field(description=…)]`
all behave exactly as they do for `@tool`.

Because `@subagent` returns a `Tool`, `load_extensions` already discovers it —
`--extension-dir` works unchanged, first-wins collision handling applies, and
the injected `midge.ext.*` logger applies. There is deliberately no agent
directory, no `--agent-dir`, and no frontmatter format. Markdown definitions
were considered and rejected: frontmatter cannot express a typed input
contract, which is the whole point of the `.py` shape.

## Binding, and why it exists

Tools cannot see the agent that called them: `Tool.invoke` receives only
schema-validated kwargs. So the entrypoint calls `bind_subagents(registry, …)`
once at startup to supply the client, model, hooks and session path. It is a
no-op on a registry with no sub-agents, so entrypoints call it unconditionally.

Threading a general context object through `Tool.invoke` for every tool was the
alternative. Rejected for now — it changes the contract for every existing tool
to serve one caller. The one field that *was* added, `call_id`, went in because
the transcript linkage genuinely needs it, not on principle.

## A child inherits tool policy and nothing else

`INHERITED_EVENTS = {"tool_call", "tool_result"}`. Those two must apply
recursively or delegation is a way around an approval policy — there is a
mutation-checked test for that regression. The other events are dropped.

Sharing everything was the first cut, and it is wrong. A hook handler is
written with the parent's conversation in mind and has no idea a child exists,
so the request-shaping events do real damage when they reach one:

- **`turn_start`** can set `system_prompt`, which silently replaces the child's
  specialised prompt. This is the one that actually bites: the override takes
  effect, nothing errors, and the sub-agent quietly stops being a sub-agent.
- **`before_provider_request`** can set `tools`, which changes what the child is
  *told* it has. Worth being precise about the blast radius: this is
  misinformation, not access. The registry gates execution independently, so a
  child that tries a tool outside its allowlist gets `Tool 'x' not found`. A
  parent hook can lie to a child; it cannot give a child tools.
- `context` and `after_provider_response` rewrite messages, and a handler
  assuming "the" conversation would be operating on the wrong one.
- `message_end` and `turn_end` cannot change behaviour but would interleave
  nondeterministically and double any turn counting.

Three events never reach a child regardless — `session_start`, `session_end`
and `before_compact` are fired by entrypoints and `compaction.compact()`, not
by `Agent`.

The filter lives in `_ChildHooks.emit` rather than at registration so it covers
`observe()` handlers too. Those see every emitted event, and since `tool_call`
still passes, an audit observer like `approval_extension`'s still records the
child's tool calls — narrowing the policy surface does not blind the audit
trail.

## Bounds

None of these existed anywhere in midge before, and delegation is what made
them necessary:

- **Wall-clock timeout** per run. The agent loop has no max-iteration bound —
  it runs until the model stops emitting tool calls — so time is what stops a
  runaway child. A `max_turns` on `Agent` would be more precise but changes the
  core loop; revisit if timeouts prove too blunt.
- **A semaphore** across concurrent spawns. N parallel tool calls already means
  N parallel sub-agents, and nothing capped that.
- **Depth**, structurally, as above.

The child is `await`ed inline rather than spawned as a task: the parent's
interrupt path *harvests* tool tasks rather than cancelling them, so a detached
child would outlive the turn that asked for it.

## Transcripts link by tool call id

A child transcript is only useful if you can say which tool call produced it.
`ToolCall.id` is already the correlation key in the format — it is on the
parent's `AssistantMessage` and again as `ToolResultMessage.tool_call_id` — so
the child is keyed on it, recorded twice:

- in the child's `SessionHeader` (`parent_session`, `parent_tool_call_id`), so
  a file found on its own says where it came from. Additive pydantic fields, so
  old sessions still load and `VERSION` stays 1.
- in the filename, `<parent-stem>.<agent>-<call-id>.jsonl`, as convenience.
  Sanitised and truncated, because a local OpenAI-compatible server can put
  anything in an id; falls back to an index when the id is unusable.

`Session` itself is unchanged — a separate file with a separate handle, so none
of the concurrent-writer hazards in `persistence.py` apply. Interleaving child
messages into the parent file was rejected: the format declares itself linear
and non-branching, and two writers on one handle corrupt it.

## Deferred

- **Nested live rendering.** The parent's history gets only the final text,
  which is the point. The user sees the ordinary tool bubble for
  `spawn_explore`. Real nesting needs a correlation id on the RPC wire and a
  tree in the TUI, and both want the envelope work in #30 first.
- **Structured results.** The child returns text; `ToolResultMessage` is always
  a single `TextContent` today.
- **Shared rate-limit awareness.** N concurrent children each run their own
  retry backoff, so a 429 storm gets N× the retries.
