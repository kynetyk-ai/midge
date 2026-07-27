from __future__ import annotations

import difflib
from itertools import pairwise

from pydantic import BaseModel

from pym.tools import tool
from pym.tools.coding._helpers import resolve_path

_BOM = "﻿"


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
            raise ValueError(f"Edit text not found: {old!r}")
        if original.find(old, idx + 1) != -1:
            raise ValueError(
                f"Edit text matches multiple locations; make it unique: {old!r}"
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
