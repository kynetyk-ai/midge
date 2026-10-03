# The RPC protocol

`midge --rpc` speaks newline-delimited JSON on stdin and stdout. This is the reference for
everything that crosses that pipe. It is pinned by `tests/test_rpc_contract.py`: the shape of every
frame listed here is recorded in `tests/golden/rpc_frames.json`, and that test fails if the code
and the record disagree — or if this document stops naming a frame type or command the code has.

**Protocol version: 1.** A client should check it (see [`ready`](#ready)) and refuse a version it
does not know.

## Compatibility

- **Not a version bump:** a new frame type, a new field on a frame or in a response's `data`, a
  new command, a new optional parameter. **Clients must ignore frame types and fields they do not
  know.**
- **A version bump:** removing or renaming a frame type, field or command; changing a field's type
  or meaning; making an optional parameter required.

midge's own version (`midge` in `ready`) says which build you are talking to. It is informational;
the protocol version is the one to check.

## Running

```bash
midge --rpc                                   # the server
python examples/rpc_client.py "say hi"        # a stdlib-only client that spawns it
```

`midge --rpc` takes the same flags as the TUI (`--session`, `--continue`, `--no-session`,
`--profile`, `--extension-dir`, `--skill-dir`, the compaction flags) and reads the same
[config](config.md). [`examples/rpc_client.py`](../examples/rpc_client.py) shows the sequence a
client follows: spawn the process, read `ready` and check `protocol`, send `prompt`, read frames
until `agent_settled`, and close stdin to finish.

midge opens no socket or port. One process serves one client, one agent and one session; running
several agents means running several processes. Logging goes to stderr or the configured log file
(see [logging](logging.md)).

## Client

`midge.rpc.client.MidgeClient` is the supported Python client; it uses only the standard library,
so a host can vendor the file.

```python
from midge.rpc.client import MidgeClient

with MidgeClient() as client:
    for frame in client.prompt("what is in README.md?"):
        if frame["type"] == "assistant_text_delta":
            print(frame["delta"], end="")
```

It implements the two rules every client needs: waiting on `agent_settled` and passing unknown
frames through.

## Framing

- One JSON object per line, UTF-8, `\n`-terminated. Output is `json.dumps(obj, ensure_ascii=False)`.
- **Stdout is the protocol and nothing else.** Diagnostics go to stderr, or to the log file.
- A line longer than **16 MiB** is refused with a [`parse`](#refusals) response and discarded; the
  next line is read normally. A blank line is ignored and unanswered.
- Stdin may be a pipe, a file or `/dev/null`; EOF ends the process after the turn in flight is cancelled and pending frames are flushed. SIGTERM and SIGHUP do the same.
- At startup the server takes the real stdout for the protocol and points `sys.stdout` at stderr,
  so a `print()` from a tool, hook or extension lands on stderr.
- Outgoing frames pass through a bounded queue. If the client stops reading and the queue fills,
  the agent pauses until the client drains it. Frames are never dropped.

## Commands and responses

A command is an object with a `type`, optional `id`, and the parameters below:

```json
{"id": "1", "type": "prompt", "message": "what is in README.md?"}
```

Every command gets exactly one `response`:

```json
{"type": "response", "command": "prompt", "success": true, "id": "1", "data": {"accepted": "started"}}
```

| Field | Present | Meaning |
|---|---|---|
| `command` | always | The command's `type`, or `parse` / `unknown` (see [Refusals](#refusals)) |
| `success` | always | Whether it was done |
| `id` | when the command had a string `id` | Echoed, for correlation |
| `data` | on success, when there is something to say | Command-specific, below |
| `error` | on failure | One line, meant to be shown to a person |

Commands are dispatched one at a time, in order. `prompt` answers as soon as the turn has started
and runs it in the background, so `abort` and `steer` can arrive while it runs.

### Running the agent

| Command | Parameters | `data` on success |
|---|---|---|
| `prompt` | `message` (string; `/skill:<name> …` expands a skill) | `accepted`: `"started"`, or `"queued"` if a turn was already running (it becomes a follow-up) |
| `steer` | `message` | `queue_id`. Injected into the running turn at its next safe point |
| `follow_up` | `message` | `queue_id`. Run as a new turn after the current one settles |
| `abort` | — | `dropped`: `[{id, content}]` — queued messages discarded. Refused if nothing is running |

A steered message is injected after every tool result of the current model response has been
recorded, before the next request is built; if the response had no tool calls, the steer starts
another request in the same run. Steering is drained at every such point and follow-ups only when
the run has nothing left to do, so pending steers are delivered before an earlier follow-up. A
`/skill:<name>` message is expanded when it is queued, and an unknown skill name is refused in the
response.

### Reading state

| Command | Parameters | `data` |
|---|---|---|
| `get_state` | — | `model`, `streaming`, `session` (path or null), `session_name`, `messages` (count), `protocol`, `midge` |
| `get_messages` | — | A **list** of messages, each as stored in the transcript |
| `get_last_assistant_text` | — | `text` (string or null) |
| `get_system_prompt` | — | `prompt` (what the model sees), `base` (the operator's part), `appended` (what extensions and skills add) |
| `get_commands` | — | `commands`: `[{name, source, invoke, description, parameters}]`. `parameters` is JSON Schema — **the machine-readable version of these tables**, including enums for `set_model` and `use_profile` |
| `get_profiles` | — | `active`, `profiles`: `[{name, description, model, tools, hooks, prompt, source}]` |
| `list_sessions` | `roots_only` (bool, default true) | `sessions`: `[{path, name, created_at, model, messages, modified, current}]` |

In `get_commands`, `invoke: "command"` means send `{"type": name, …}` with `parameters` as keys of
the object; `invoke: "prompt"` (skills, named `skill:<name>`) means send the text `/<name> …` in a
`prompt`, `steer` or `follow_up` message. An empty `properties` object means the entry takes no
arguments.

With `roots_only`, `list_sessions` omits transcripts that another transcript started: sub-agent
runs, profile excursions and forks (see [sub-agents](subagents.md) and [sessions](sessions.md)).

### Changing the agent

The ones marked **idle** are refused while a turn is running, because they replace history or tools
the turn is using; the others take effect from the next request.

| Command | Parameters | `data` on success |
|---|---|---|
| `set_model` | `model` | `durable` (whether it was recorded in the transcript) |
| `set_system_prompt` | `prompt` | `durable` |
| `compact` · **idle** | — | `summary`, `cut_index`, `message_count`. `summary` and `cut_index` are null if there was nothing to summarize |
| `clear_context` · **idle** | — | `cleared` (messages dropped from context), `session`. The transcript keeps them |
| `new_session` | `path` | `session` |
| `open_session` · **idle** | `path` | `session`, `reopened`, `name`, `messages`, `model`, `recorded_model`, `model_differs` |
| `set_session_name` | `name` | `name` |
| `use_profile` · **idle** | `name`, `transcript` (`continue` · `fork` · `resume_last`, default `continue`) | `profile`, `requested` and `transcript` (the transcript mode asked for and the one used), `model`, `tools`, `session`, `messages`, `durable` |
| `reload` · **idle** | `targets` (list of `skills` / `extensions`; default both) | `targets`, `tools`, `skills`, `profiles` (counts after reloading) |

`open_session` restores the transcript's history and base system prompt and keeps the running
model. `recorded_model` is the model the transcript last recorded, and `model_differs` says whether
it differs from `model`, so a client can offer `set_model`.

### Refusals

`success: false` with an `error`. The process always keeps serving.

- **`parse`** — the line was not JSON, not an object, or longer than the limit. There is no `id`
  to echo.
- **`unknown`** — `type` missing or not a string. An unrecognised string `type` is echoed as
  `command` with `unknown command: '<type>'`.
- **`id` not a string** — refused with ``"`id` must be a string"``, and the command is not run.
- **A bad parameter** — one line naming it: ``"`transcript`: Input should be 'continue', 'fork' or
  'resume_last'"``.
- **Not allowed now** — e.g. `cannot compact while a run is in flight`, `no prompt in flight`.
- **The operation failed** — e.g. a `compact` whose summary request failed: `"<Error>: <message>"`.

## Events

Frames with no `command`: what the agent is doing. A client should switch on `type` and ignore
types it does not know.

### `ready`

```json
{"type": "ready", "protocol": 1, "midge": "0.1.0"}
```

**The first frame, always**, before any command is read. A client that cannot speak `protocol`
should close stdin and stop.

### During a turn

In roughly the order they occur:

| Frame | Fields | When |
|---|---|---|
| `user_message` | `content`; plus `source: "steer"` and `queue_id` for a steered message | The turn's opening message, and each steered message as it is injected |
| `assistant_text_delta` | `delta` | Streamed text |
| `tool_call_start` | `id`, `name` | The model began a tool call |
| `tool_call_delta` | `id`, `delta` | Its arguments, streamed as JSON text |
| `tool_call_end` | `id`, `name`, `arguments` (object) | The call is complete |
| `assistant_message_end` | `stop_reason`, `model` | One model response finished (`stop`, `tool_use`, `length`, …) |
| `tool_execution_start` | `id`, `name` | A tool began running |
| `tool_result` | `tool_call_id`, `content` (text), `is_error` | It finished. A denied or blocked call is an `is_error` result |
| `error` | `message`, `stop_reason` (`error` or `aborted`) | The turn failed or was aborted |
| `agent_end` | — | One run finished; a follow-up may start another |
| `compaction_start` | — | History crossed `[compaction] threshold`; summarizing |
| `compaction_end` | `cut_index`, `message_count`; `error` if the summary failed | Summarizing done |
| `queue_update` | `steering`, `follow_up`: each `[{id, content}]` | The queues changed |
| `agent_settled` | — | **Terminal.** Nothing more will happen for this prompt |

Guarantees a client can rely on:

- **`agent_settled` is last**, and always arrives — on success, error and abort alike. Wait on it,
  not on `agent_end`: a queued follow-up makes one prompt produce several `agent_end`s.
- `compaction_start` / `compaction_end` may arrive during a turn, between a `tool_result` and the
    next model output, or after `agent_end`. They always come in pairs and before `agent_settled`.
- Tool calls in one model response that change state run one at a time, in order; read-only ones
  may overlap. Each `tool_result` carries the `tool_call_id` it answers.

### From a sub-agent

A frame produced by a nested agent (a `spawn_*` tool) carries an `agent` envelope:

```json
{"type": "tool_execution_start", "id": "c9", "name": "read",
 "agent": {"agent": "explore", "agent_id": "call_1", "parent_id": null, "depth": 1}}
```

`agent_id` is the id of the tool call that spawned the child — the same id its transcript records
as `parent_tool_call_id`. Only tool executions, errors and the child's end are forwarded; its text
is in its own transcript. A frame with no `agent` key is the top-level agent's.

## Persistence

With a session (the default, or `--session PATH`), each turn is written to the transcript as it
ends — or, if it is aborted or fails, as far as it got. A new process opening the same file (with
`--session`, `--continue` or `open_session`) resumes the conversation. See [sessions](sessions.md) for the transcript format.

## Trust

**Anything that can write a line to stdin can run `bash` with this process's privileges.** There is
no authentication and no approval prompt in RPC mode; the boundary is the container or OS user
midge runs as. `src/midge/rpc/transport.py` records what a bridge to a socket would have to decide
for itself.
