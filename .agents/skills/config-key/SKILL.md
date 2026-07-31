---
name: config-key
description: >-
  Adds a new setting to midge's config surface, or audits an existing one. Use
  when a value needs to be tunable — a timeout, a limit, a path, a threshold, a
  model id — or when reviewing a change that introduced a constant, an
  os.getenv, or an argparse default.
license: MIT
---

# Adding a config key

`src/midge/config.py` owns every setting. A tunable that lives anywhere else is
findable only by grepping the source, which is the failure this file exists to
prevent.

## Decide whether it *is* a config key

It is, if the value would plausibly differ between two people, two machines, or
two deployments. A timeout, a limit, a threshold, a path, a level, a model id.

It is not, if the number follows from the code rather than the deployment. The
`[:64]` filename cap and the `range(100)` collision retry in `subagents.py` are
deliberately *not* config: they want a comment explaining the number, not a
knob. Do not file a key to avoid explaining a constant.

**Never a credential.** `api_key` in a config file gets committed. A provider
names the *variable* holding its key (`api_key_env`), never the key.

## The three edits

Doing two of the three leaves the key unreachable, so make all three together.

1. **The field**, on `Config` or the right nested dataclass (`RetryConfig`,
   `SessionConfig`, `SubagentConfig`, `LogConfig`), with its default.

2. **The line in `Config.load`**, using the accessor that matches the type:
   `text`, `integer`, `number`, `flag`, `path`, `choice`. Pass an env var name
   as the third positional argument only if the setting genuinely wants one —
   `[retry]` and `[subagents]` deliberately have none.

3. **The entry in `examples/config.toml`**, in the right section, with prose
   above it saying *why* rather than what. Uncomment it only if the value shown
   equals the default.

## Why the third edit is not optional

`tests/test_config.py::test_the_shipped_example_parses_with_no_diagnostics`
loads `examples/config.toml` and asserts it round-trips to `Config()` with no
diagnostics. A key in the example that no accessor reads is reported as unknown;
a key read but not documented drifts silently. Either way CI fails, which is the
point.

Run it:

```bash
poetry run pytest tests/test_config.py -q
```

## Reaching the value

Config goes to **entrypoints only** and is passed inward. There is deliberately
no `get_config()`.

The trap worth knowing: an argparse `default=` that is not `None` makes the
config layer unreachable, because the flag always wins. Precedence is
**flag > env > config > default**, and the default belongs on `Config`, once.

A library default in a signature (`Client(max_attempts=3)`) is fine, but it is
not a substitute — the entrypoint must still pass the configured value or
nobody can reach it.

## Bad input

A malformed file, an unknown key or a wrongly typed value degrades to the
default and records a `Diagnostic`. It never raises: a typo must not stop the
harness from starting, and must not silently change what it does either.

`load` parses and logs nothing — it returns diagnostics the entrypoint passes to
`emit` after `logs.configure`, because the log level is one of the things it
resolves.

## Check the work

```bash
poetry run pytest tests/test_config.py -q
poetry run ruff check . && poetry run pyright
grep -rn "os.getenv" src/midge/ | grep -v config.py
```

That last one should return only the two credential reads —
`providers/openai_compat.py` and `providers/registry.py`. A third is the thing
to push back on.
