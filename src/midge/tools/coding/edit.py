from __future__ import annotations

import difflib
from itertools import pairwise

from pydantic import BaseModel

from midge.tools import tool
from midge.tools.coding._helpers import resolve_path

_BOM = "﻿"

# Below this, the closest region is too unlike `old_text` to be the one meant,
# and naming it would send the model after the wrong text.
_CLOSE_ENOUGH = 0.6
# How much of the file's actual text a hint quotes back.
_QUOTE_CHARS = 1500


class EditOp(BaseModel):
    old_text: str
    new_text: str


@tool
async def edit(path: str, edits: list[EditOp]) -> str:
    """Apply one or more text replacements to a file. Each edit's `old_text` must
    match exactly once in the file. All edits are matched against the original
    content, so they cannot overlap. Returns a unified diff and the 1-indexed
    first changed line.

    Preserves UTF-8 BOM and CRLF line endings if present.
    """
    if not edits:
        raise ValueError("edits must not be empty")

    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if p.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path}")

    raw = p.read_bytes().decode("utf-8")
    has_bom = raw.startswith(_BOM)
    if has_bom:
        raw = raw.removeprefix(_BOM)
    has_crlf = "\r\n" in raw
    original = raw.replace("\r\n", "\n") if has_crlf else raw

    ranges: list[tuple[int, int, str]] = []
    for op in edits:
        old = op.old_text.replace("\r\n", "\n") if has_crlf else op.old_text
        new = op.new_text.replace("\r\n", "\n") if has_crlf else op.new_text
        idx = original.find(old)
        if idx == -1:
            raise ValueError(_not_found(original, old))
        if original.find(old, idx + 1) != -1:
            lines = _match_lines(original, old)
            raise ValueError(
                f"Edit text matches multiple locations (lines {', '.join(map(str, lines))}); "
                "include more surrounding text to make it unique."
            )
        ranges.append((idx, idx + len(old), new))

    ranges.sort(key=lambda r: r[0])
    for (_s1, e1, _n1), (s2, _e2, _n2) in pairwise(ranges):
        if e1 > s2:
            raise ValueError("Overlapping edits are not allowed")

    parts: list[str] = []
    cursor = 0
    for s, e, nt in ranges:
        parts.append(original[cursor:s])
        parts.append(nt)
        cursor = e
    parts.append(original[cursor:])
    result = "".join(parts)

    out = result.replace("\n", "\r\n") if has_crlf else result
    if has_bom:
        out = _BOM + out

    p.write_bytes(out.encode("utf-8"))

    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            result.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    first_changed = _first_changed_line(original, result)
    return f"first_changed_line: {first_changed}\n{diff}"


def _first_changed_line(original: str, result: str) -> int:
    a = original.splitlines()
    b = result.splitlines()
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i + 1
    return min(len(a), len(b)) + 1


# --- when old_text does not match ------------------------------------------
#
# Every hint below *describes* the file; none of them applies anything. An exact
# match is what proves the model is editing the text it believes it is, so a
# near miss stays a failure — just one that says what to fix (#97).


def _not_found(original: str, old: str) -> str:
    file_lines = original.split("\n")
    want = old.strip("\n").split("\n")
    windows = [
        (i, file_lines[i : i + len(want)]) for i in range(len(file_lines) - len(want) + 1)
    ]

    squash = [" ".join(line.split()) for line in want]
    for i, window in windows:
        if _same_lines([" ".join(line.split()) for line in window], squash):
            return (
                "Edit text not found exactly, but it matches "
                f"{_span(i, len(want))} if whitespace is ignored — the indentation "
                f"or spacing differs. The file has:\n{_quote(window)}"
            )

    # Scored as how much of what was *sent* appears in the window, in order,
    # rather than symmetric similarity: a one-line near miss inside a longer
    # line is still the line meant, and would otherwise score as "nothing close".
    best, best_at = 0.0, -1
    target = "\n".join(want)
    for i, window in windows:
        m = difflib.SequenceMatcher(None, "\n".join(window), target, autojunk=False)
        found = sum(block.size for block in m.get_matching_blocks()) / max(len(target), 1)
        if found > best:
            best, best_at = found, i
    if best >= _CLOSE_ENOUGH:
        window = file_lines[best_at : best_at + len(want)]
        first = best_at + 1 + next(
            (k for k in range(len(want)) if not _line_matches(k, len(want), window[k], want[k])),
            0,
        )
        return (
            f"Edit text not found. The closest match is {_span(best_at, len(want))} "
            f"({best:.0%} of it found there), first differing at line {first}. The file has:\n"
            f"{_quote(window)}"
        )
    return (
        "Edit text not found, and nothing in the file is close to it — "
        "this may be the wrong file, or content that has changed since it was read."
    )


def _line_matches(k: int, n: int, have: str, sent: str) -> bool:
    # `old_text` rarely starts at a line boundary or ends at one, so its first
    # line is compared as a suffix of the file's, its last as a prefix, and a
    # single line as a substring. Only the lines in between must be whole.
    if n == 1:
        return sent in have
    if k == 0:
        return have.endswith(sent)
    if k == n - 1:
        return have.startswith(sent)
    return have == sent


def _same_lines(have: list[str], sent: list[str]) -> bool:
    return all(_line_matches(k, len(sent), h, w) for k, (h, w) in enumerate(zip(have, sent, strict=True)))


def _match_lines(original: str, old: str) -> list[int]:
    lines, at = [], original.find(old)
    while at != -1:
        lines.append(original.count("\n", 0, at) + 1)
        at = original.find(old, at + 1)
    return lines


def _span(start: int, count: int) -> str:
    return f"line {start + 1}" if count == 1 else f"lines {start + 1}-{start + count}"


def _quote(lines: list[str]) -> str:
    text = "\n".join(lines)
    return text if len(text) <= _QUOTE_CHARS else text[:_QUOTE_CHARS] + "…"
