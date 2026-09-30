# Sessions

Every run records its conversation to an append-only JSONL transcript, and a transcript can be reopened to resume the conversation. The code is [`src/midge/persistence.py`](../src/midge/persistence.py).

## Where transcripts go

Recording is on by default. Without `--session`, each run creates a timestamped file in the session directory, `.midge/sessions/` under the working directory unless `[session] dir` says otherwise.

| Flag or key | Effect |
|---|---|
| `--session PATH` | Use this file, resuming it if it exists. A relative path is taken under the session directory; an absolute one is used as given. |
| `--continue` | Resume the most recently modified session in the directory, or start a new one if there is none |
| `--no-session` | Record nothing for this run |
| `[session] enabled = false` | Record nothing by default; an explicit `--session` still records |

A running agent can also switch transcripts with the `new_session` and `open_session` commands, name one with `set_session_name`, and list them with `list_sessions` (see [rpc](rpc.md)).

## When turns are written

`Controls.run_turn` in [`src/midge/commands.py`](../src/midge/commands.py) writes what reached history. A completed run is written from `agent_end`'s `new_messages`. A run that is cancelled or fails is written from the history it produced, which the [agent loop](agent-loop.md) has already closed off with results for any unanswered tool calls. Each line is flushed as it is written.

## Format

The first line is the header. Every other line is a message or a record.

| `type` | Contains | Meaning |
|---|---|---|
| `header` | `version`, `created_at`, `model`, `system_prompt`, and optionally `origin`, `parent_session`, `parent_tool_call_id` | What the session started as |
| `message` | `data`: a `Message` as JSON | One history entry |
| `compaction` | `summary`, `cut_index` | Replace the first `cut_index` messages with the summary ([compaction](compaction.md)) |
| `clear` | `cut_index` | Drop the first `cut_index` messages from context |
| `session_info` | `name` | The display name |
| `model_change` | `model` | The model was switched |
| `identity` | `system_prompt` | The base system prompt was replaced |
| `profile` | `name`, `model`, `system_prompt` | The agent was switched to a [profile](profiles.md) |
| `continued` | `path`, `reason` | Another transcript of this session started here |

Records carry a millisecond `timestamp`.

The header is never rewritten. Anything that changes during a session is an appended record, and loading replays them in order with the last one winning. Read `Session.model`, `Session.system_prompt`, `Session.name` and `Session.profile` for the current values; `Session.header` is only the origin.

`clear` and `compaction` leave the original messages in the file. They change what a resume puts back in context, and anything reading the file directly still sees the full record.

`identity` and `profile` store the base prompt only. The skills catalogue and extension additions are recomposed on every start from what is installed at that time.

## Loading

`read_transcript` returns the header and every entry; `fold_history` replays the records into the history the agent held. Loading:

- rejects a file with no header, or whose `version` is newer than this build's; older versions load;
- drops a truncated final line, which is what a crash during a write leaves, and raises on a corrupt line anywhere else;
- skips an entry type it does not recognise, with a warning.

The version number moves only when an older build skipping a new record would misread the session.

Tools and extensions are not stored. A resumed session uses whatever the current process loads; a call to a tool that no longer exists fails with an error result.

## Resuming

A resumed session restores its history, base system prompt and profile. The model is restored differently: a model set explicitly for this run (`MIDGE_MODEL` or `model` in a config file) wins, with a warning if it differs from the recorded one; otherwise the recorded model is used. With a [model registry](providers.md#the-model-registry), a recorded model that is no longer listed falls back to the configured one with a warning.

The resume logic is `resume_identity` in [`src/midge/cli.py`](../src/midge/cli.py).

## Sessions across several files

A [sub-agent](subagents.md) writes its own transcript, and a profile switch with `transcript="fork"` starts a new one. These files sit beside the parent and link both ways:

- the child's header sets `origin` (`subagent` or `profile`) and `parent_session`, and a sub-agent's also sets `parent_tool_call_id`, the id of the tool call that spawned it;
- the parent appends a `continued` record whose `path` is the child's file name, relative to the parent's directory.

`session_chain(path)` walks up to the root and back down through `continued` records to list every transcript of a session. `list_sessions` and `--continue` consider only root transcripts, since reopening a child would resume the middle of a tool call.

## Using it directly

`Session.new`, `Session.load` and `Session.open` (load if the file exists, otherwise create) return a session open for appending. `append`, `append_many`, `append_compaction`, `append_clear`, `set_name`, `set_model`, `set_system_prompt` and `set_profile` each write one line and update the in-memory view to match what a reload would build. [`examples/coding_agent.py`](../examples/coding_agent.py) shows the wiring.

Only one process may have a session file open at a time.
