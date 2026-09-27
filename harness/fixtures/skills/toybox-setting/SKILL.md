---
name: toybox-setting
description: >-
  Adds a setting to toybox, or audits one that already exists. Use when a value
  needs to be tunable — a width, a path, a limit, a flag — or when reviewing a
  change that introduced a bare constant or an environment variable read.
license: MIT
---

# Adding a toybox setting

`src/toybox/settings.py` owns every tunable. A value that lives anywhere else is
findable only by grepping, which is what this file exists to prevent.

## Decide whether it is one

It is, if the value would differ between two machines or two runs — a width, a
path, a limit, a flag.

It is not, if the number follows from the code. `truncate` reserving one
character for the ellipsis is arithmetic, not configuration.

## The three edits

Doing two of the three leaves the setting unreachable, so make all three.

1. **The field**, on `Settings`, with its default and a comment saying what it
   is for.
2. **The line in `load`**, reading it out of the parsed TOML with the right
   coercion, degrading to the default rather than raising.
3. **The entry in `settings.example.toml`**, with prose above it saying why.

Read `references/checklist.md` in this skill's directory for the exact order and
the mistakes that keep recurring.

## Check the work

```bash
python -m pytest -q
python -c "from toybox.settings import load; print(load())"
```

The second one matters more than it looks: a field added without its line in
`load` still constructs, still has a default, and silently ignores the file.
