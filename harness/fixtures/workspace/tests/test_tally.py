from toybox.tally import average_length, most_common, word_counts


def test_word_counts_folds_case_by_default() -> None:
    assert word_counts("A a b") == {"a": 2, "b": 1}


def test_word_counts_can_respect_case() -> None:
    assert word_counts("A a b", case_sensitive=True) == {"A": 1, "a": 1, "b": 1}


def test_most_common_is_ordered() -> None:
    assert most_common("a a a b b c", n=2) == [("a", 3), ("b", 2)]


def test_average_length_of_nothing_is_zero() -> None:
    assert average_length([]) == 0.0


def test_average_length() -> None:
    assert average_length(["ab", "cdef"]) == 3.0
