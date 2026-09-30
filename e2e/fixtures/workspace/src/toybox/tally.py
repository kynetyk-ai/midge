"""Counting. One function here does not do what its docstring says."""

from __future__ import annotations

from collections import Counter


def word_counts(text: str, case_sensitive: bool = False) -> dict[str, int]:
    """How many times each word appears."""
    words = text.split() if case_sensitive else text.lower().split()
    return dict(Counter(words))


def most_common(text: str, n: int = 3) -> list[tuple[str, int]]:
    """The `n` most frequent words, most frequent first."""
    return Counter(text.lower().split()).most_common(n)


def average_length(words: list[str]) -> float:
    """The mean length of the given words.

    Returns 0.0 for an empty list rather than raising.
    """
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)
