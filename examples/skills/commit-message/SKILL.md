---
name: commit-message
description: >-
  Writes a git commit message for the staged changes in this repository,
  following the project's conventions. Use when the user asks to commit, to
  write or rewrite a commit message, or to describe a change for history.
license: MIT
---

# Commit message

## Gather the change

Run these before writing anything — a message written without reading the diff
describes what you assumed, not what changed.

```bash
git diff --cached --stat
git diff --cached
git log --oneline -10
```

If nothing is staged, say so and stop rather than guessing from the working tree.

## Write it

Read `references/style.md` in this skill's directory for the full rules and
worked examples. In short:

- One line of subject, then a blank line, then prose.
- Subject is `type(scope): imperative summary`, lower case, no trailing period.
- The body explains *why*, and what the reader would otherwise have to
  reconstruct. Skip what the diff already shows.
- Wrap the body at 80 columns.

## Hand it over

Print the message in a fenced block and stop. Do not run `git commit` unless the
user asks — a message they have not read is not a message they approved.
