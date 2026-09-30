# Skills

A skill is a directory holding a `SKILL.md` file in the [Agent Skills](https://agentskills.io/specification) format: instructions for one kind of task, written in markdown with no Python. midge lists each skill's name and description in the system prompt, and the model opens the file when a task matches. The implementation is [`src/midge/skills.py`](../src/midge/skills.py).

Skills teach the agent to use tools it already has. To add a capability, write an [extension](extensions.md).

## Writing one

`SKILL.md` starts with YAML frontmatter carrying a `description` and usually a `name`, followed by the instructions. Other files beside it (`references/`, `scripts/`, `assets/`) are freeform.

```
my-skills/
└── commit-message/
    ├── SKILL.md
    └── references/
        └── style.md
```

```markdown
---
name: commit-message
description: >-
  Writes a git commit message for the staged changes, following the project's
  conventions. Use when the user asks to commit or to describe a change.
---

# Commit message

Run `git diff --cached` first. Read `references/style.md` for the full rules.
```

```bash
midge --skill-dir my-skills
```

[`examples/skills/commit-message/`](../examples/skills/commit-message/) is a worked example.

## Discovery

Skills are loaded from an ordered list of sources:

1. each `--skill-dir`, in command-line order;
2. `.midge/skills` and `.agents/skills` under the working directory;
3. `~/.midge/skills` and `~/.agents/skills`;
4. `[skills] system_dir` from the config file (see [`examples/config.toml`](../examples/config.toml)).

Items 2 to 4 come from `default_skill_dirs()`. A source that does not exist is logged and skipped. When two skills share a name, the one from the earlier source wins and the other is dropped with a `skill_name_shadowed` warning. The order of this list decides precedence, never the order files happen to be found in.

Within a directory, the walk recurses into subdirectories and stops at any directory that contains `SKILL.md`, so a skill's own `references/` tree is never read as nested skills. It skips dot-directories, `node_modules` and `__pycache__`, and gives up below a fixed depth so that `--skill-dir ~` does not crawl a home directory. A source may also name a single `.md` file. The same file reached twice through different sources or symlinks is loaded once.

## Validation

Validation is lenient so that directories written for other harnesses load unchanged; `--skill-dir ~/.claude/skills` works.

- A missing or empty `description` skips the skill, since the description is the only thing the model sees before deciding to open the file. Missing or unparseable frontmatter also skips it.
- `name` defaults to the directory name. A name that is too long, uses characters outside `a-z`, `0-9` and `-`, or has leading, trailing or doubled hyphens logs a warning and still loads. A name need not match its directory.
- An over-long description logs a warning and still loads.
- `license`, `compatibility`, `metadata` and `allowed-tools` are accepted and ignored. To restrict tools, use a [hook](hooks.md) or a [profile](profiles.md).

Every problem is a `skill_*` warning on the `midge.skills` logger. A bad skill never stops startup.

## The catalogue

`skills_prompt()` renders an `<available_skills>` block with each skill's name, description and absolute path, preceded by an instruction to open a skill's file with the reader tool when a task matches and to resolve relative paths against the skill's directory. midge appends it to the system prompt after the base prompt and any extension contributions. The skill body is never loaded at startup, so a long reference document costs nothing until the model reads it.

The catalogue appears only when some registered tool declares `reads_paths` (`path_reader()` returns its name). The built-in `read` does. An agent without such a tool, for example under a profile that omits `read`, gets no catalogue, since it would be told to use a tool it does not have; its skills remain reachable by explicit invocation. See [tools](tools.md#reads_paths).

The catalogue lives in the system prompt, which compaction does not touch. A skill body the model has read is ordinary history and can be summarized away; the model can read it again from the path in the catalogue (see [compaction](compaction.md)). midge recomposes the catalogue on every start and after `reload`, so skills added to disk appear without starting a new session.

Add `disable-model-invocation: true` to a skill's frontmatter to leave it out of the catalogue. Explicit invocation still works.

## Invoking a skill explicitly

A model does not always open a matching skill. Sending `/skill:<name>` as the message forces one:

```
/skill:commit-message keep it to one line
```

midge reads `SKILL.md` at that moment, wraps its body in a `<skill name=… location=…>` block that names the skill's directory, and sends it as the user message, with any text after the name appended as further instructions. This works in the TUI input box and in RPC `prompt`, `steer` and `follow_up` messages (see [rpc](rpc.md)). An unknown name is refused. Every skill, including one with `disable-model-invocation`, appears as `skill:<name>` in the TUI command palette and in the RPC `get_commands` listing.

A forced body is an ordinary user message and is subject to compaction like any other.
