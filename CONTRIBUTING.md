# Contributing

Thanks for your interest. This is a small personal-scale project — the core harness is roughly 3.5k LOC, deliberately readable and hackable, and most contributions probably take the form of bug fixes, new extensions, or small focused features. Before opening a PR with significant scope, please open an issue first to align on direction.

## Dev setup

```bash
poetry install
poetry run pytest          # run the suite
poetry run ruff check      # lint
poetry run pyright         # type-check
```

Python 3.11+ is required. `poetry.lock` is committed; please use Poetry rather than `pip`/`uv`/`pip-tools` so dependency state stays consistent.

## Style and conventions

The codebase optimizes for clarity over cleverness. A few load-bearing conventions:

- Prefer three similar lines to a premature abstraction.
- Default to no docstrings unless the function is part of a public API.
- Comment the **why**, not the **what** — well-named identifiers handle the latter.
- No backwards-compatibility scaffolding for code that does not yet exist.
- Keep abstractions cheap.
- Log through `logging.getLogger(__name__)`, never `print()`. Message format is a `snake_case` event name then `key=%s` pairs, so errors stay countable. Only entrypoints configure logging; see the Logging section of [`CLAUDE.md`](./CLAUDE.md).

[`CLAUDE.md`](./CLAUDE.md) captures more of the project's "feel" and is worth a skim before a substantive change.

## Tests and types

- Every new feature or bug fix should land with at least one test.
- `pyright` and `ruff` must be clean on `main`. PRs that introduce new findings will be asked to fix them.
- Public-API surface (anything imported in an `examples/` file or in the README's "Layout" section) should have type annotations.

## Adding an extension

The fastest way to extend the harness is to write an extension. See [`examples/notes_extension/`](./examples/notes_extension/) for a complete example. In short:

1. Create a `.py` file with `@tool`-decorated async functions.
2. Optionally export a module-level `SYSTEM_PROMPT` string.
3. Run with `--extension-dir path/to/your/dir`.

If you'd like the extension upstreamed as a built-in, open an issue describing the use case.

## Pull requests

- Branch off `main`.
- One logical change per PR; small PRs are easier to review and revert.
- Commit messages: imperative mood, scope prefix when natural (`feat(extensions):`, `fix(rpc):`, `docs:`). The history follows that pattern.
- CI runs `ruff`, `pyright`, and `pytest`; please run them locally before pushing.

## Reporting a bug

Use a GitHub issue. Include:
- A minimal reproduction (ideally a failing test).
- The Python version and (if relevant) the OpenAI-compatible endpoint you're hitting.
- The full error / output, not a paraphrase.

For security-sensitive reports, see [`SECURITY.md`](./SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the project's MIT license — see [`LICENSE`](./LICENSE).
