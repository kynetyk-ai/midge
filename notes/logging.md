# Logging — design notes

Implemented in `src/midge/logs.py`. Unlike the other notes here this one is not
borrowed from `pi-mono`; `pi` configures logging through its own devtools and
settings machinery, none of which midge has. What follows is why midge's version
looks the way it does.

The user-facing conventions live in the Logging section of `CLAUDE.md`. This
note records the reasoning behind them.

## The constraint that shapes everything: three modes, three meanings of "stderr"

| mode | stdout | stderr |
|---|---|---|
| RPC (`examples/rpc_agent.py`) | **the protocol** — one stray byte breaks every client | free |
| headless (`examples/coding_agent.py`) | the rendered transcript | free |
| TUI (`cli.py`, `examples/notes_agent.py`) | captured by Textual and dropped | **unsafe for an eagerly-bound handler** |

The TUI case is the one that bites. `textual/app.py` wraps its message loop in
`redirect_stdout`/`redirect_stderr`, which rebind the `sys.stdout`/`sys.stderr`
*names*. But `logging.StreamHandler.__init__` resolves and **stores** its stream
at construction. A handler built in `cli.main()` — fifty lines before
`App.run()` — therefore holds the real terminal stream, writes past the
redirect, and shreds the display.

Before this change midge was safe only by accident: with nothing configured,
records went to `logging.lastResort`, whose `stream` is a *property* re-read at
emit time, so it followed the redirect and got swallowed. Correct output, for
entirely the wrong reason, and only at WARNING.

`textual.logging.TextualHandler` resolves the active app per record via the
`active_app` ContextVar, so it is safe to construct at any point. Its catch is
that it routes to `app.log.logging(...)` — the devtools console, invisible
without `textual console`. Hence `[log] file`: in the TUI it is the practical
way to read logs, not a convenience.

**The rule that falls out: libraries never configure logging, only entrypoints
do**, because only the entrypoint knows the mode. `tui_log_handler(log_file)`
lives in `tui/app.py` rather than `logs.py` so the core does not need to know
Textual exists — and it takes the path rather than reading the environment,
because that is `config.py`'s job now. Configuration is parsed before logging is
configured (the log level is one of the things it resolves), which is why
`config.load` returns diagnostics instead of logging them.

## `propagate` stays True

Not a preference — a hard constraint. Every `caplog` test in the suite does
`caplog.at_level(logging.WARNING, logger="midge.x")`, but pytest installs its
capturing handler on the **root** logger. Records only reach it by propagation.
Setting `propagate = False` on `midge` would not fail those tests loudly; it
would empty `caplog.records` and make them fail on assertions that look
unrelated. `tests/test_logs.py` asserts the propagation guarantee directly so
the next person to reach for `propagate = False` gets a pointed failure.

## Payloads at DEBUG, and the one thing that is never logged

The original issue said "log shapes and outcomes, never payloads", on the
grounds that the session JSONL and the RPC event stream already record
everything. That is true of *messages*, and false of everything else. The
assembled `create_kwargs`, the raw stream chunks, and the malformed tool-call
buffer exist nowhere but the log, and they are exactly what is needed when a
local or quantized model behind `base_url` misbehaves — midge's primary target.
So payloads are logged at DEBUG, truncated via `payload()`.

Credentials are a different category and are excluded at every level. An
`api_key` never appears at all; a `base_url` goes through `provider_host()`,
which keeps the hostname and discards userinfo and query string, both of which
can carry secrets. `client.py` stores neither on `self`, so the hostname has to
be logged by the entrypoint that reads the environment variable.

The `openai` logger keeps its own `openai_level` (env `MIDGE_LOG_LEVEL_OPENAI`) for the same reason.
The original justification — that the SDK dumps request bodies at DEBUG — no
longer disqualifies it, since midge now logs those itself in truncated form.
What remains is that the SDK's DEBUG output can carry the `Authorization`
header.

## Why `payload()` returns an object rather than a string

`logging` only formats arguments once a handler will emit the record, but
`payload(x)` as an argument is evaluated regardless of level. Returning a
string would make every DEBUG payload cost a full `repr()` and truncation even
at WARNING. Deferring into `__str__` moves that cost to where it belongs. The
same trap applies to any other expensive argument — keep them O(1) or wrap
them.

## Event names over prose

The goal is counting occurrences without a metrics subsystem, so a message is a
`snake_case` identity followed by `key=%s` pairs, and `grep -c bash_timeout`
just works. The codebase had drifted into two styles — structured in the newer
modules, prose in `extensions.py`, `hooks.py` and all of `skills.py` — so all 34
sites were converted at once rather than documenting a rule half the code
violated. ruff's `G` and `LOG` rulesets now enforce the mechanical half.

The trade-off is that the tests match on message substrings, so renaming an
event breaks a test. That is the right coupling: an event name is an interface.

## Deferred

- **JSON-vs-text formatting by `isatty()`**, and container ergonomics generally
  (`PYTHONUNBUFFERED`, stdout/stderr discipline under `docker logs`). Worth
  doing when there is a Dockerfile; there isn't one.
- **`run_id` / `turn_id` correlation** via `ContextVar` — roughly ten lines, but
  only useful once there is enough INFO traffic to need correlating, and
  `SessionHeader` has no id field to hang it on today.
- **A SIGTERM handler.** Belongs with the container work. Note `bash.py` uses
  `start_new_session=True`, so tool subprocesses orphan on SIGTERM to PID 1, and
  Python as PID 1 does not reap zombies.
- **Per-module level env vars.** `logging.getLogger("midge.client").setLevel(DEBUG)`
  in a REPL or a three-line extension already covers anyone who needs this.
- **Structured diagnostics returned to callers** rather than logged — see the
  departures table in `notes/skills.md`. midge has no consumer for them.

Explicitly not doing: `structlog`, `loguru`, `python-json-logger`,
OpenTelemetry; a `dictConfig`/YAML/TOML config file; a custom `Logger` subclass
or `LoggerAdapter` hierarchy; `QueueHandler`/`QueueListener`; rotation or
retention; a `--verbose` flag duplicated across four argparse blocks — one env
var works identically in a shell, a Dockerfile, and `docker run`.
