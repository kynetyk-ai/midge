# The container harness

A disposable midge, driven over RPC from outside the container, so every
extension mechanism can be exercised against a real model.

```bash
python3 harness/midgectl.py up --build
python3 harness/midgectl.py call get_state
python3 harness/midgectl.py prompt "read README.md and say what toybox is"
python3 harness/midgectl.py logs      # midge.log, stderr, docker logs
python3 harness/midgectl.py down
```

`up` recreates the container, which is also how you reset: the workspace is
re-copied from the image and re-committed, so an agent that mangled a file
leaves nothing behind.

Extra arguments after `up` reach `midge --rpc`:

```bash
python3 harness/midgectl.py up --extension-dir /opt/midge/examples/approval_extension
python3 harness/midgectl.py up --skill-dir /opt/harness/skills
```

## Driving the TUI by hand

The RPC harness is blind to anything that only exists on a screen. For that:

```bash
python3 harness/midgectl.py tui        # prints a `docker run -it` line
```

It prints rather than runs, because Textual needs a TTY this process does not
have. See [TUI-PROTOCOL.md](./TUI-PROTOCOL.md) for what to exercise, and
[TUI-WORKSHEET.md](./TUI-WORKSHEET.md) to write down what you saw — including
the one place the two front-ends genuinely differ (#99: the TUI writes messages
to the transcript, RPC does not).

## Why the FIFO

`serve_stdio` shuts down on stdin EOF, and every `docker exec` is a separate
process — so `echo … | docker exec -i` would run one command and kill the
server. The entrypoint holds the FIFO open with `sleep infinity` so the
last-writer-closed condition never arrives. Output goes to a file rather than a
second FIFO, so reads come from a byte offset and nothing is lost between calls.

## What is inside

| path | what |
|---|---|
| `/work` | the toybox workspace, a git repo, re-copied on every start |
| `/work/.midge/config.toml` | small limits, so timeouts and compaction are reachable |
| `/opt/midge/examples` | midge's shipped extensions, loaded per scenario |
| `/opt/harness/skills` | the harness's own skill, for toybox rather than midge |
| `/run/midge/{in,out,err,midge.log}` | the control channel and the evidence |

`OPENAI_API_KEY` arrives via `--env-file .env` at run time. Nothing in `src/`
reads a `.env`, so baking one into the image would do nothing.
