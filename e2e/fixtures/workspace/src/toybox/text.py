"""String helpers, with the edge cases that make them worth testing."""

from __future__ import annotations


def wrap(text: str, width: int = 72) -> list[str]:
    """Break `text` into lines no longer than `width`.

    A word longer than `width` gets a line of its own rather than being split —
    breaking an identifier in half is worse than an over-long line.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    lines: list[str] = []
    current = ""
    for word in text.split():
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def slugify(title: str) -> str:
    """A filesystem-safe form of `title`."""
    out = "".join(c.lower() if c.isalnum() else "-" for c in title)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def truncate(text: str, limit: int) -> str:
    """Cut `text` to `limit` characters, marking that it was cut."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
