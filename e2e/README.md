# End-to-end tests

A disposable midge, driven over RPC from outside the container, so every
extension mechanism can be exercised against a real model.

```bash
python3 e2e/midgectl.py up --build
python3 e2e/midgectl.py call get_state
python3 e2e/midgectl.py prompt "read README.md and say what toybox is"
python3 e2e/midgectl.py logs      # midge.log, stderr, docker logs
python3 e2e/midgectl.py down
```

`up` recreates the container, which is also how you reset: the workspace is
re-copied from the image and re-committed, so an agent that mangled a file
leaves nothing behind.

Extra arguments after `up` reach `midge --rpc`:

```bash
python3 e2e/midgectl.py up --extension-dir /opt/midge/examples/approval_extension
python3 e2e/midgectl.py up --skill-dir /opt/e2e/skills
```

RPC transcripts are mounted to `e2e/.state/rpc-sessions/` on the host, so
they survive `up` — which is what makes "restart, then `open_session`" a thing
you can test rather than a thing the container forgets.

## Driving the TUI

The RPC scenarios are blind to anything that only exists on a screen. Two ways in.

**Scripted, through tmux** (`brew install tmux`) — how an agent or a script
checks the TUI, since neither owns a terminal:

```bash
python3 e2e/midgectl.py tui-up -- --extension-dir /opt/midge/examples/approval_extension
python3 e2e/midgectl.py tui-type "read README.md"
python3 e2e/midgectl.py tui-keys Enter          # tmux key names: C-c, Escape, M-Enter…
python3 e2e/midgectl.py tui-screen --until "toybox" --timeout 60
python3 e2e/midgectl.py tui-down
```

`tui-screen` prints the *rendered* screen as text, so a check asserts on what a
person would see — a tag Textual swallowed as markup is visibly missing.

**By hand:**

```bash
python3 e2e/midgectl.py tui        # prints a `docker run -it` line
```

See [TUI-PROTOCOL.md](./TUI-PROTOCOL.md) for what to exercise, and
[TUI-WORKSHEET.md](./TUI-WORKSHEET.md) to write down what you saw — including
the one place the two front-ends genuinely differ (#99: the TUI writes messages
to the transcript, RPC does not).

## The merge gate

A PR into `develop` that changes observable behaviour gets a **behavioral
test** here before it merges: RPC checks scripted under `scenarios/`, TUI checks
driven through tmux, each recorded in the PR body as PASS / FAIL / ODD with the
evidence. Unit tests are necessary and not sufficient — #99 shipped with a green
suite, because nothing drove a turn through the server and then read the disk.

What a real model cannot be made to do on demand (an empty assistant turn, say)
is marked *unit-only* rather than faked.

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
| `/opt/e2e/skills` | the e2e tests' own skill, for toybox rather than midge |
| `/run/midge/{in,out,err,midge.log}` | the control channel and the evidence |

`OPENAI_API_KEY` arrives via `--env-file .env` at run time. Nothing in `src/`
reads a `.env`, so baking one into the image would do nothing.
