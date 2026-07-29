#!/usr/bin/env python3
"""Enforce a size budget on the core harness.

    poetry run python scripts/loc.py

The core is the top-level modules in `src/midge/` — the harness itself. It
deliberately excludes `tools/` and `tui/`: built-in tools are a library that
grows with the domain, and the TUI is a presentation layer. Neither is the
thing the README claims you can read in a sitting.

Prose is free. Blank lines, `#` comments and docstrings are all subtracted, so
the number measures machinery. That is the point: an author who hits the cap
should be pushed to simplify the code, not to delete the explanation of why it
works that way.

A bare string statement counts as prose (it is a block comment); a string
assigned to a name counts as code (it is data — an HTML template, a system
prompt).

Exits non-zero when the core is over budget.
"""

from __future__ import annotations

import io
import sys
import tokenize
from pathlib import Path

BUDGET = 5_000

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "src" / "midge"
# Reported but not enforced, so code moving out of the core to duck the budget
# is visible rather than silent.
CONTEXT = (CORE / "tools", CORE / "tui", ROOT / "tests")


def prose_lines(source: str) -> set[int]:
    """1-indexed lines that are blank, a `#` comment, or a docstring."""
    out = {i for i, line in enumerate(source.splitlines(), 1) if not line.strip()}
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    for i, tok in enumerate(tokens):
        if tok.type == tokenize.COMMENT:
            out.add(tok.start[0])
        elif tok.type == tokenize.STRING and _is_statement(tokens, i):
            out.update(range(tok.start[0], tok.end[0] + 1))
    return out


def _is_statement(tokens: list[tokenize.TokenInfo], i: int) -> bool:
    """True when the string at `i` stands alone — a docstring, not an operand."""
    for prev in reversed(tokens[:i]):
        if prev.type in (tokenize.COMMENT, tokenize.NL):
            continue
        return prev.type in (tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE)
    return True  # first token in the file


def count(path: Path) -> int:
    source = path.read_text(encoding="utf-8")
    return len(source.splitlines()) - len(prose_lines(source))


def total(directory: Path, *, recursive: bool = False) -> int:
    pattern = "**/*.py" if recursive else "*.py"
    return sum(count(p) for p in sorted(directory.glob(pattern)))


def main() -> int:
    files = sorted(CORE.glob("*.py"), key=count, reverse=True)
    counted = sum(count(p) for p in files)

    context = [(f"{d.name}/", total(d, recursive=True)) for d in CONTEXT]
    width = max(len(label) for label in [p.name for p in files] + [c[0] for c in context])

    for p in files:
        print(f"  {p.name:<{width}}  {count(p):>5}")
    print(f"  {'':<{width}}  {'-' * 5}")
    print(f"  {'core':<{width}}  {counted:>5}  / {BUDGET}")

    print("\n  not counted:")
    for label, n in context:
        print(f"  {label:<{width}}  {n:>5}")

    print()
    if counted > BUDGET:
        print(f"OVER BUDGET by {counted - BUDGET} lines.")
        print("Simplify, or move what is not core into tools/ or an extension.")
        return 1
    print(f"Within budget. {BUDGET - counted} lines of headroom.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
