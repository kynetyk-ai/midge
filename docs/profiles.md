# Profiles

A profile is a named configuration the agent runs as: a system prompt, a model, a subset of the discovered tools and a decision for each hook source. Selecting one applies all four at once. The reasoning behind the design is in [ADR 0001](adr/0001-session-profiles.md).

Code: [`src/midge/profiles.py`](../src/midge/profiles.py) (declaration and validation) and `Controls.use_profile` / `Controls.apply_profile` in [`src/midge/commands.py`](../src/midge/commands.py) (applying one). Example: [`examples/profile_extension/`](../examples/profile_extension/).

## Declaring a profile

A profile is a `Profile` instance at module level in an extension `.py` file. The file may also declare tools and sub-agents; `--extension-dir` discovers all of them, and there is no separate profile directory or flag.

```python
from midge.profiles import Profile

REVIEWER = Profile(
    name="adversarial-reviewer",
    description="Reviews recent work, looking for what is wrong with it.",
    prompt="Assume the work is wrong and find out how. Cite `path:line` for every claim.",
    tools=("read", "ls", "grep"),
    hooks={"approve": True},
)
```

| Field | Meaning |
|---|---|
| `name`, `description` | How the profile is selected and listed. |
| `prompt` | The base system prompt. It replaces `[agent] system_prompt` and the built-in default. |
| `model` | The model to switch to. Empty (the default) keeps whatever model the agent is running. |
| `tools` | The tools the agent has under this profile, by name. Discovered tools not listed here are unavailable. |
| `hooks` | `True` or `False` for every discovered hook source, keyed by the extension file's stem (`approve.py` is `"approve"`). |

`tools` may name a `spawn_*` tool. Sub-agents are re-bound to the profile's tool set, so a sub-agent cannot use a tool the profile excludes.

## Validation

Profiles are validated once, after every extension has loaded, because a profile may name a tool from another file. A profile with any of these problems is dropped with a warning:

| Problem | Event |
|---|---|
| Empty `name` or `prompt` | `profile_invalid` |
| A tool that was not discovered | `profile_tool_unknown` |
| A hook source that was not discovered | `profile_hook_unknown` |
| A discovered hook source missing from `hooks` | `profile_hook_undecided` |
| A model not in a non-empty model registry | `profile_model_unregistered` |

`hooks` must decide every discovered hook source. Leaving a tool out of `tools` gives a less capable agent; leaving a hook out would give an unguarded one, so omission is an error. Adding a new hook-bearing extension therefore invalidates each existing profile until it names the new source.

## Selecting a profile

At startup, the first of these that is set wins:

1. `--profile NAME`
2. the profile recorded in a resumed session
3. `[profiles] default` in the config file (or `MIDGE_PROFILE`)

With none set, midge runs under no profile. A name that does not match a usable profile stops startup with an error, which distinguishes a profile that was never discovered from one that failed validation. When a resumed session's recorded profile is no longer discovered, midge warns (`resume_profile_unavailable`) and uses the session's recorded prompt and model.

```bash
poetry run midge --extension-dir examples/profile_extension \
  --extension-dir examples/approval_extension --profile adversarial-reviewer
```

## Switching at runtime

`use_profile` switches profiles while the agent is idle and is refused during a turn. It is available over [RPC](rpc.md) (with `get_profiles` to list what is available) and in the TUI drawer's profile list, which always uses the `continue` transcript mode described below. Either every dimension changes or none does: anything that can fail, such as an unknown name or a transcript that will not open, fails before anything is applied. The switch is recorded in the session, so every message can be traced to the profile that produced it.

There is no revert command; to go back, switch to the other profile by name. Conversation history is kept across a switch. Send `clear_context` afterwards for a clean slate.

### Transcripts

The `transcript` argument decides which file the following turns are written to (see [sessions](sessions.md)):

| Value | Effect |
|---|---|
| `continue` (default) | Keep writing to the current transcript. |
| `fork` | Open a new transcript linked to the current one. History in memory is unchanged. |
| `resume_last` | Reopen this session's most recent transcript that was last running under the named profile, and load its history. |

```jsonc
{"type":"use_profile","name":"reviewer","transcript":"fork"}         // step out
{"type":"use_profile","name":"builder","transcript":"resume_last"}   // and back
```

`resume_last` searches only the current session's chain of linked transcripts and skips sub-agent transcripts. When the profile has no earlier transcript there, the switch still succeeds using `[profiles] resume_fallback` (`fork` by default, or `continue`). A `fork` with no session to fork from degrades to `continue` with a warning. The response reports both the mode requested and the mode used.

## Profiles and sub-agents

A profile and a [sub-agent](subagents.md) share several fields, and are kept separate on purpose. A profile is what the agent is, applied by an operator or client. A sub-agent is a `spawn_*` tool the agent calls to delegate work to a nested agent, and it returns a result to its caller.

See [retargeting](retargeting.md) for using a profile, an extension and config to point midge at a new domain.
