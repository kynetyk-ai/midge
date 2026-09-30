from toybox.text import slugify, truncate, wrap


def test_wrap_keeps_lines_within_width() -> None:
    lines = wrap("the quick brown fox jumps over the lazy dog", width=12)
    assert all(len(line) <= 12 for line in lines)


def test_a_word_longer_than_the_width_gets_its_own_line() -> None:
    lines = wrap("a supercalifragilistic b", width=5)
    assert "supercalifragilistic" in lines


def test_wrap_rejects_a_nonsense_width() -> None:
    try:
        wrap("x", width=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_slugify_collapses_runs() -> None:
    assert slugify("Hello,   World!!") == "hello-world"


def test_truncate_marks_the_cut() -> None:
    assert truncate("abcdef", 4) == "abc…"
    assert truncate("abc", 10) == "abc"
