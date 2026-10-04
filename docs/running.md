# Running midge

midge runs as an interactive TUI, as an [RPC](rpc.md) server, in a container, or in a Docker Sandbox. Every mode uses the same `midge` entrypoint, [configuration](config.md) and session files.

midge works on the directory it starts in: the coding tools read, write and run commands relative to it, and [sessions](sessions.md) are saved under its `.midge/sessions/`. Start it from the project you want to work on.

## Install

The Quick start [in the README](../README.md#quick-start) shows how to install from a release tag. For development:

```bash
poetry install                               # from this repo
export OPENAI_API_KEY=sk-...
```

midge reads the key from the environment only and does not load `.env` files. The model and provider come from [configuration](config.md); a missing key is reported when the TUI opens. Without the `tui` extra only `midge --rpc` is available; add `[tui]` for the TUI.

`poetry run midge` works only inside this repo, because Poetry runs it from there. To start midge in another project, call the installed script:

```bash
alias midge="$(poetry -C ~/coding/midge env info -p)/bin/midge"
cd ~/code/my-project
midge
```

## Flags

| Flag | Effect |
|---|---|
| `--extension-dir DIR` | Load [extensions](extensions.md) from `DIR` (repeatable) |
| `--skill-dir DIR` | Load [skills](skills.md) from `DIR` (repeatable) |
| `--profile NAME` | Start under a [profile](profiles.md) |
| `--rpc` | Serve the [RPC protocol](rpc.md) on stdin/stdout instead of opening the TUI |
| `--session PATH`, `--continue`, `--no-session` | Choose, resume, or skip the transcript (see [sessions](sessions.md)) |
| `--compaction-threshold N`, `--compaction-keep-recent N` | Override the [compaction](compaction.md) settings |
| `--version` | Print the version |

## The TUI

Type a request and press `Enter`.

| Key | Action |
|---|---|
| `Enter` | Submit |
| `Ctrl+O` | Newline (`Alt+Enter` also works where the terminal sends Option as Meta) |
| `Ctrl+P` | Command palette: things you do (compact, clear, reload, skills) |
| `Ctrl+B` | Drawer: the model, profile and session, and switching between them |
| `Ctrl+C` | Interrupt the turn and drop anything queued |
| `Ctrl+D` | Quit |
| `Esc` | Close the drawer or clear the input |

**Commands.** The TUI and the RPC server call the same `Controls` object and list the same command table, so each offers everything the other does. `Ctrl+P` lists the commands that take no argument (`compact`, `clear_context`, `reload`, `abort`) and each skill. A leading slash reaches every command and can carry an argument: `/compact`, `/set_model gpt-4o`, `/open_session prior.jsonl`, `/skill:review`. A slash is intercepted only when the word after it is a real command, so `/etc/hosts is missing` is sent as a message.

**The drawer.** `Ctrl+B` opens a panel listing the models, profiles and sessions, with the current one of each marked. `Up`/`Down` move through all three sections, `Tab`/`Shift+Tab` jump between sections, `Enter` switches to the highlighted entry, and `Esc` closes the panel. A section with nothing to offer says why; the model section lists alternatives only when a `[models]` table defines them (see [config](config.md#the-model-registry)).

**Tool approval.** The TUI asks before running any tool that is not `read_only`: `bash`, `write`, `edit`, an extension's tool, or a sub-agent that may call one (and then each call that sub-agent makes). `y` allows once, `a` allows that tool for the rest of the session, and `n` refuses and tells the model. `[tui] approve_tools = false` turns this off. RPC never asks, because an unattended caller has nobody to answer; in RPC mode the boundary is the container.

**Queued messages.** Typing while a turn runs queues the message. It is delivered at the next tool boundary, so work already done is kept. `Ctrl+C` is the explicit interrupt.

## RPC

```bash
midge --rpc
```

The protocol, a minimal client and embedding guidance are in [rpc](rpc.md).

## In a container

The root `Dockerfile` builds the midge wheel from source and installs it into an image that works on whatever is mounted at `/workspace`. The agent sees that directory and nothing else of the host, so the container bounds what `bash` can reach.

```bash
docker build -t midge .                     # on Linux: --build-arg UID=$(id -u) --build-arg GID=$(id -g)
cd ~/code/my-project
docker run -it --rm --detach-keys ctrl-^ \
  -e OPENAI_API_KEY -v "$PWD":/workspace midge
```

- Arguments after the image name go to `midge`: `... midge --continue`, or `... midge --rpc` (with `-i` and without `-t`).
- `--detach-keys ctrl-^` is required for the TUI. Docker's default detach sequence starts with `Ctrl+P` and consumes it before midge sees it.
- Sessions are written to the project's `.midge/sessions/`, so they persist on the host. The project's `.midge/config.toml` is read as usual; to use your personal one as well, add `-v ~/.midge:/home/midge/.midge:ro`.
- The `UID`/`GID` build arguments make files the agent writes owned by you. Docker Desktop on macOS maps ownership itself and does not need them.
- The image has `git`, `ripgrep`, `curl`, `jq` and `make`. For a project that needs its own toolchain, build on it: `FROM midge`, install what it needs as `root`, and switch back to `USER midge`.

## In a Docker Sandbox

[`sandbox/midge/`](../sandbox/midge/) is a kit for [Docker Sandboxes](https://docs.docker.com/ai/sandboxes/) (`sbx`). midge runs in a microVM with the project mounted, outbound traffic limited to an allowlist, and the OpenAI key added to requests by a host proxy, so the real key never enters the sandbox. The image adds uv, Poetry, conda, Node and a C toolchain. The allowlist and resource limits are in [`midge.yaml`](../sandbox/midge/midge.yaml).

Store the key once, then run the kit on a project:

```bash
sbx secret set openai                        # or `sbx secret import openai` to take OPENAI_API_KEY
sbx run --name midge-myproj ~/coding/midge/sandbox/midge ~/code/my-project
sbx run --name midge-myproj                  # re-attach later
```

- The project is mounted, so the agent's edits appear on the host immediately.
- Arguments after `--` go to `midge`: `sbx run --name midge-myproj -- --continue`.
- The image installs midge from GitHub at the tag or commit in `args.ref.default` (a release tag by default). `sbx` rebuilds the image only when the kit directory changes, and `sbx run --kit-arg` cannot set `ref` because it is resolved at build time.
- midge starts in the project, so its `.midge/config.toml` applies (see [config](config.md)).
- The kit's `sandbox-env` skill, which tells the agent how to use the sandbox's package managers and network, loads from `[skills] system_dir`. Your host's shared skills store is mounted at `~/.agents/skills`.
- `sbx ls` lists sandboxes, `sbx stop NAME` stops one and `sbx rm NAME` deletes it.

With `--clone`, the agent works on a private clone of the host repository and cannot touch your working tree. Its branches are fetched from a `sandbox-NAME` remote that `sbx` adds to the host repository ([Docker's git workflow](https://docs.docker.com/ai/sandboxes/workflows/git/)):

```bash
cd ~/code/my-project
sbx run --clone --name midge-safe ~/coding/midge/sandbox/midge
git fetch sandbox-midge-safe
git ls-remote sandbox-midge-safe             # branch names, which must match exactly
git checkout -b my-branch sandbox-midge-safe/my-branch
git push -u origin my-branch
```

The remote is reachable only while the sandbox is running and is removed by `sbx rm`, so push anything worth keeping first.

## One-shot CLI

[`examples/coding_agent.py`](../examples/coding_agent.py) runs one prompt and prints the transcript, without the TUI:

```bash
poetry run python -m examples.coding_agent "list files in cwd"

OPENAI_BASE_URL=http://127.0.0.1:1234/v1 MIDGE_MODEL=ibm/granite-3.2-8b \
  poetry run python -m examples.coding_agent --session run.jsonl "summarize the README"
```

It takes `--extension-dir`, `--skill-dir`, `--skill NAME` (send a skill as the turn), `--session`, `--no-session`, `--compaction-threshold` and `--compaction-keep-recent`.

## A second domain

```bash
midge --extension-dir examples/notes_extension --profile notes
```

The same TUI, RPC server and sessions, running as a knowledge assistant: the `notes` profile in [`examples/notes_extension/notes.py`](../examples/notes_extension/notes.py) gives it its own identity and only the note tools, with no shell or file editing. The knowledge base is `~/.midge-notes/kb.json` unless `MIDGE_NOTES_KB` names another file. To make it a project's default, set `[extensions] enabled = true` and `[profiles] default = "notes"` in its `.midge/config.toml`. [Retargeting](retargeting.md) builds a domain like this from an empty directory.
