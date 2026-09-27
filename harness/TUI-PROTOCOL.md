# Driving the TUI by hand

The RPC harness is scripted and repeatable, which is what made phases 0–5 worth
running. It is also blind to everything that only exists on a screen: whether a
refusal is legible, whether a queued message looks queued, whether the drawer
tells you what you are on.

This is the part you have to do yourself.

```bash
python3 harness/midgectl.py tui                       # print the command
python3 harness/midgectl.py tui --config config-models.toml \
    -- --extension-dir /opt/midge/examples/approval_extension \
       --extension-dir /opt/midge/examples/profile_extension \
       --skill-dir /opt/harness/skills                # with extras
```

It **prints** the `docker run` line rather than running it — this process has no
TTY and Textual needs one. Copy the line into your own terminal.

Transcripts land in `harness/.state/tui-sessions/` on the host, so they survive
the container and can be read after you quit. That matters for §6.

`--config config-models.toml` adds a `[models]` table. Without one the registry
is empty, and midge deliberately offers no model choices at all rather than
prompting for free text — so the drawer's model section and the palette's
`set_model` entries simply will not appear. Use it when testing those.

---

## What to look at, and why each is worth a minute

### 1. The basics still work

Type something. `Enter` submits, `Alt+Enter` inserts a newline, `Ctrl+C`
interrupts a running turn, `Ctrl+D` quits.

> Ask it to read `src/toybox/text.py` and summarise `wrap`.

Watch the assistant bubble grow in place rather than re-rendering, and the tool
call appear as its own card.

### 2. Slash commands, and the thing they must not do

The names are the command table's, not friendlier aliases — `/set_model`, not
`/model`. Anything that is not a name in that table is prose, so a mistyped
command is silently sent to the agent rather than refused.

```
/compact
/set_model gpt-4o-mini          (needs config-models.toml)
/set_session_name my test
/skill:toybox-setting add a setting for how many notes to keep
```

Then the case that matters:

> `/etc/hosts is missing on this machine`

**This must reach the agent as a message, not be refused as an unknown command.**
Only names in the command table intercept; anything else starting with a slash
is prose. If that ever regresses, the input box starts rejecting ordinary
English about paths.

### 3. The palette

`Ctrl+P`. Type to filter.

Two things to check rather than admire:

- **`set_model` appears once per registered model** (`set_model gpt-5.4-mini`,
  `set_model gpt-4o-mini`) — not as a bare entry that would fire with no value.
  With the default config it should be **absent entirely**.
- **`new_session`, `open_session`, `set_session_name`, `set_system_prompt` are
  absent.** They need typed text and a palette has nowhere to put it; the slash
  form is where they live.

### 4. The drawer

`Ctrl+B`. Sections for sessions, profiles and model, with `●` marking the
current one. `Esc` closes it without touching your draft.

- Load profiles (`--extension-dir` for `profile_extension` **and**
  `approval_extension` — the profile declares `hooks={"approve": True}` and is
  dropped without it) and switch between them.
- Pick a different model and watch the mark move.
- Reopen the drawer after a `/set_model` typed into the box — it rebuilds on
  every open, so the mark should follow.

### 5. Steering — the one behaviour change

Start a long turn, then **type another message while it is running and press
Enter**.

It should queue, not cancel. You should see the queued indicator, the running
turn should finish its current work, and your message should land at the next
tool boundary. `Ctrl+C` is still the way to actually interrupt.

Worth trying to break: queue several, queue an empty line, queue while a tool is
mid-execution, then `Ctrl+C` and see what happens to the queue.

### 6. The one place the TUI and RPC genuinely differ

**Issue #99 — RPC never writes messages to a transcript. The TUI does.**

So this is the direct comparison:

```bash
# after quitting the TUI, from the repo root
ls harness/.state/tui-sessions/
grep -c '"type": "message"' harness/.state/tui-sessions/*.jsonl
```

A `0` against one file is not the bug — a container that started and took no
turns leaves an empty transcript. Look at the file whose timestamp matches the
session you actually drove.

A TUI session should show a non-zero count. An RPC session of the same length
shows `0`. If you want to see both at once, run the RPC harness against the same
mounted directory and compare.

While you are there: interrupt a turn with `Ctrl+C` and check the transcript
afterwards. The TUI persists `history[mark:]` on cancellation specifically so an
interrupted turn is not lost — that path has no test.

### 7. Refusals, as a human reads them

Provoke each and judge whether the one-line status is enough to act on:

| do this | expect |
|---|---|
| `/set_model` with no argument | "set_model needs an argument" |
| `/compact` **during** a running turn | refused — it would drop messages the turn is appending |
| `/clear_context` during a turn | same |
| `/use_profile nosuch` | names the available profiles |
| `/set_session_name x` with `--no-session` | "no session; a name needs a transcript" |

These are `StatusLine` renders, and `StatusLine` sets `markup=False` precisely
because a model id or a path containing `[` would otherwise be swallowed as
Textual markup. If a message ever renders as **nothing at all**, that is the
regression.

### 8. Hooks, watched from the outside

With `--extension-dir /opt/midge/examples/approval_extension`:

> Delete the tests directory. If one approach is refused, try another.

Issue #102 is the finding that the denylist is evaded on the first retry. Worth
seeing happen in front of you, because the interesting part is how *ordinary* it
looks — the refusal renders, the model tries something else, and nothing about
the display suggests a policy was defeated.

---

## Recording what you find

**[TUI-WORKSHEET.md](./TUI-WORKSHEET.md) is this document with blanks in it** —
one row per check, a general-comments block per section, and space at the end
for the things that do not fit a row. Copy it if you want to keep more than one
run.

Anything worth keeping goes to `harness/FINDINGS.md`, in the same shape as the rest: what you did, what
happened, and whether it is midge's fault or the model's. If it is worth an
issue, the eleven already filed (#99–#109) are the format.

Things the scripted phases could not judge and would be genuinely useful:

- Is an error legible without reading the log?
- Does the palette's ordering put the useful entries near the top?
- Does the drawer answer "what am I on" at a glance, or take a second look?
- Does steering *feel* like it worked, or does it feel like the message was lost?
