# Logging

midge logs through the standard `logging` module under the `midge` logger tree. The entrypoint
installs one handler, chosen by mode, and every other module only acquires a logger. The code is
[`src/midge/logs.py`](../src/midge/logs.py).

## Configuring it

The `[log]` section of the [config](config.md) file sets four keys, each with an environment
variable:

| Key | Env | Default | Effect |
|---|---|---|---|
| `level` | `MIDGE_LOG_LEVEL` | `WARNING` | Level for the `midge` loggers, including extensions |
| `openai_level` | `MIDGE_LOG_LEVEL_OPENAI` | `WARNING` | Level for the `openai` SDK logger |
| `file` | `MIDGE_LOG_FILE` | unset | Write logs to this file instead of the mode's default |
| `payload_chars` | `MIDGE_LOG_PAYLOAD_CHARS` | `2000` | Truncation length for DEBUG payloads; `0` disables truncation |

See [`examples/config.toml`](../examples/config.toml) for the documented entries.

An unknown level name falls back to `WARNING` and logs `log_level_invalid`. A log file whose
directory is missing is created with its parents; a file that cannot be opened falls back to the
mode's default handler and logs `log_file_unusable`. `httpx` is held at `WARNING` regardless of
configuration.

`openai_level` is separate from `level` because the SDK's DEBUG output can include the
`Authorization` header. Raising `level` to `DEBUG` leaves the SDK at its own level.

## Where logs go

Only entrypoints call `logs.configure`, because the correct destination depends on what stdout and
stderr mean in each mode:

| Mode | stdout | Default log destination |
|---|---|---|
| RPC (`midge --rpc`) | The [protocol](rpc.md) | stderr |
| Headless ([`examples/coding_agent.py`](../examples/coding_agent.py)) | The rendered transcript | stderr |
| TUI (`midge`) | Owned by Textual | Textual's devtools console |

In the TUI, a `logging.StreamHandler` binds `sys.stderr` when it is constructed and so writes past
Textual's redirect onto the screen. `tui_log_handler` in [`src/midge/tui/app.py`](../src/midge/tui/app.py)
returns Textual's `TextualHandler`, which resolves the running app per record. Its output appears
only under `textual console`, so setting `[log] file` is the practical way to read logs from the
TUI. When the file cannot be opened, the TUI falls back to `TextualHandler`, never stderr.

Configuration is parsed before logging is configured, because the log level is one of the values
it resolves. `Config.load` therefore returns diagnostics, and the entrypoint emits them once the
handler is installed.

## Message format

Each record is formatted as `%(asctime)s %(levelname)-7s %(name)s %(message)s`. The message starts
with a `snake_case` event name followed by `key=value` pairs:

```
2026-01-01 12:00:00,000 WARNING midge.skills skill_description_missing path=/x/SKILL.md
```

Occurrences of an event can be counted with `grep -c <event>`. Tests match on event names, so
renaming one is an interface change.

Levels: **ERROR** the operation failed; **WARNING** degraded but continuing; **INFO** the
operational narrative; **DEBUG** why it did that.

## Payloads and credentials

Request bodies, tool arguments, tool results, `bash` commands and compaction summaries are logged
at DEBUG only, wrapped in `logs.payload()`. It renders `repr(value)` and truncates it to
`payload_chars`. The rendering happens in `__str__`, so a payload costs nothing unless a handler
emits the record.

Credentials are never logged at any level. An API key never appears. A `base_url` is logged through
`logs.provider_host()`, which keeps only the hostname because the userinfo and query string can
carry secrets.

## Adding logging

A module declares `_logger = logging.getLogger(__name__)` at the top and never configures logging
itself. `getLogger(__name__)` is what places the logger under `midge`, gives it a per-module name in
the output, and makes it visible to pytest's `caplog`; a wrapper or adapter in front of it breaks
those. A module run with `python -m` has `__name__ == "__main__"`, which is outside the `midge`
tree, so name its logger explicitly (as `examples/coding_agent.py` does).

Extension files receive a `log` attribute bound to `midge.ext.<file stem>`, so `[log] level`
covers them too. See [extensions](extensions.md).

The rules the code relies on:

- Use lazy `%s` arguments; no f-string in the format string. ruff's `G` and `LOG` rules enforce
  this.
- Keep arguments O(1), because they are evaluated whether or not the level is enabled. Wrap
  anything expensive in `logs.payload()`.
- An `except` that swallows an exception logs it, with `exc_info=e` when the traceback would
  otherwise be lost.
- Do not set `propagate = False` on the `midge` logger. pytest's `caplog` handler sits on the root
  logger, and records reach it only by propagation. `tests/test_logs.py` checks this.
- Nothing may write to stdout in RPC mode. `midge --rpc` redirects `sys.stdout` to stderr at
  startup so a stray `print()` cannot corrupt the protocol, but a logging handler must still never
  target stdout.
