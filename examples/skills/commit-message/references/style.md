# Commit message style

Loaded on demand by the `commit-message` skill. Nothing here is in the model's
context until it opens this file — that is the point of splitting it out.

## Subject

```
type(scope): imperative summary
```

`type` is one of:

| type | when |
|---|---|
| `feat` | a new capability a user can reach |
| `fix` | a defect repaired |
| `refactor` | behaviour unchanged, structure changed |
| `perf` | measurably faster or smaller |
| `test` | tests only |
| `docs` | documentation only |
| `deps` | dependency bumps |

`scope` is optional and names the subsystem, not the file: `agent`, `client`,
`export`, `messages`, `tools`.

Rules:

- Imperative mood: "add", not "adds" or "added".
- No capital letter after the colon, no trailing period.
- 72 characters or fewer. If it will not fit, the commit is probably two commits.

## Body

The subject says what. The body says why, and only what the diff cannot.

Good:

```
fix(client): retry retryable provider failures with a cancellable backoff

A 429 or a 5xx killed the whole turn, which meant a transient blip cost the
user their context. The retry sleeps on an event rather than the clock so an
interrupt during the backoff is still honoured immediately.
```

Bad — restates the diff:

```
fix(client): add retry loop

Added a for loop around the request with a sleep between attempts. Also added
a max_retries parameter and a constant for the base delay.
```

## Trailers

Leave a blank line, then trailers, one per line:

```
Closes #39
Co-Authored-By: Name <email>
```

Use `Closes #N` only when the commit genuinely resolves the whole issue.
