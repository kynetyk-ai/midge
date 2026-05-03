from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import pytest

from pi.client import Client
from pi.compaction import (
    COMPACTION_PREFIX,
    COMPACTION_SUFFIX,
    compact,
    count_tokens,
    find_cut_index,
    make_summary_message,
    needs_compaction,
    summarize,
)
from pi.messages import (
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def _chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


class _FakeStream:
    def __init__(self, chunks: Iterable[Any]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> Any:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _install_turns(client: Client, turns: list[list[Any]]) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    iterator = iter(turns)

    async def create(**kwargs: Any) -> _FakeStream:
        captured.append(kwargs)
        return _FakeStream(next(iterator))

    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return captured


# ---- count_tokens ----

def test_count_tokens_empty_history() -> None:
    assert count_tokens([]) == 0


def test_count_tokens_grows_with_content() -> None:
    short = [UserMessage(content="hi")]
    long = [UserMessage(content="hi " * 1000)]
    assert count_tokens(long) > count_tokens(short) * 5


# ---- find_cut_index ----

def test_find_cut_index_empty_history() -> None:
    assert find_cut_index([], keep_recent_tokens=1000) == 0


def test_find_cut_index_no_user_messages_returns_zero() -> None:
    history: list[Message] = [
        AssistantMessage(content=[TextContent(text="weird")]),
    ]
    assert find_cut_index(history, keep_recent_tokens=10_000) == 0


def test_find_cut_index_everything_fits_returns_zero() -> None:
    history: list[Message] = [
        UserMessage(content="hi"),
        AssistantMessage(content=[TextContent(text="hello")]),
    ]
    assert find_cut_index(history, keep_recent_tokens=10_000) == 0


def test_find_cut_index_lands_on_user_boundary() -> None:
    """Each turn = (user, assistant). With three turns and a budget that fits
    only the last turn, the cut should be at index 4 (start of turn 3)."""
    history: list[Message] = []
    for i in range(3):
        history.append(UserMessage(content=f"q{i}: " + "x" * 200))
        history.append(AssistantMessage(content=[TextContent(text="a" * 200)]))
    # Tokens-per-turn under our estimator is roughly proportional to char count.
    # Budget that fits ~one turn:
    one_turn = count_tokens(history[-2:])
    cut = find_cut_index(history, keep_recent_tokens=one_turn + 5)
    assert cut == 4
    assert isinstance(history[cut], UserMessage)


def test_find_cut_index_keeps_tool_pair_with_assistant() -> None:
    """The cut must NOT split a ToolCall from its ToolResultMessage."""
    history: list[Message] = [
        UserMessage(content="x" * 500),
        AssistantMessage(content=[TextContent(text="a" * 500)]),
        UserMessage(content="now use a tool"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="read", arguments={"path": "p"})],
            stop_reason="tool_use",
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="read",
            content=[TextContent(text="...")],
        ),
        AssistantMessage(content=[TextContent(text="ok")]),
    ]
    # Budget large enough to keep turn 2 (msgs 2..5):
    suffix = count_tokens(history[2:])
    cut = find_cut_index(history, keep_recent_tokens=suffix + 5)
    assert cut == 2
    assert isinstance(history[cut], UserMessage)
    # Verify no tool result is now orphaned in the kept suffix
    assert history[cut + 1].content[0].name == "read"  # type: ignore[union-attr]
    assert isinstance(history[cut + 2], ToolResultMessage)


def test_find_cut_index_oversized_last_turn_kept_anyway() -> None:
    """If even the most recent turn exceeds the budget, keep it; don't try to
    split it."""
    history: list[Message] = [
        UserMessage(content="x" * 500),
        AssistantMessage(content=[TextContent(text="a" * 500)]),
        UserMessage(content="y" * 5000),
        AssistantMessage(content=[TextContent(text="b" * 5000)]),
    ]
    cut = find_cut_index(history, keep_recent_tokens=100)
    assert cut == 2
    assert isinstance(history[cut], UserMessage)


def test_find_cut_index_picks_earliest_fitting_user_boundary() -> None:
    """When several user-boundary suffixes fit, prefer the earliest (compact
    as much as possible)."""
    history: list[Message] = []
    for i in range(4):
        history.append(UserMessage(content=f"q{i}"))
        history.append(AssistantMessage(content=[TextContent(text=f"a{i}")]))
    cut = find_cut_index(history, keep_recent_tokens=10_000_000)
    # Everything fits → cut == 0 (no compaction needed)
    assert cut == 0


# ---- needs_compaction ----

def test_needs_compaction_below_threshold() -> None:
    assert (
        needs_compaction([UserMessage(content="hi")], threshold_tokens=10_000)
        is False
    )


def test_needs_compaction_above_threshold() -> None:
    big: list[Message] = [UserMessage(content="x" * 100_000)]
    assert needs_compaction(big, threshold_tokens=100) is True


# ---- make_summary_message ----

def test_make_summary_message_wraps_text() -> None:
    m = make_summary_message("## Goal\nFoo")
    assert isinstance(m, UserMessage)
    assert isinstance(m.content, str)
    assert m.content.startswith(COMPACTION_PREFIX)
    assert m.content.endswith(COMPACTION_SUFFIX)
    assert "## Goal\nFoo" in m.content


# ---- summarize ----

async def test_summarize_collects_text_deltas() -> None:
    client = Client()
    captured = _install_turns(
        client,
        [
            [
                _chunk(content="## Goal\n"),
                _chunk(content="learn things"),
                _chunk(finish_reason="stop"),
            ]
        ],
    )
    text = await summarize(
        client,
        "m",
        [UserMessage(content="hi"), AssistantMessage(content=[TextContent(text="hey")])],
    )
    assert text == "## Goal\nlearn things"

    # Verify the system prompt was set to the summarization prompt
    sys_msg = captured[0]["messages"][0]
    assert sys_msg["role"] == "system"
    assert "summarization assistant" in sys_msg["content"]
    # Last user message should be the SUMMARIZE_INSTRUCTION
    last = captured[0]["messages"][-1]
    assert last["role"] == "user"
    assert "summary" in last["content"].lower()


async def test_summarize_empty_prefix_raises() -> None:
    client = Client()
    with pytest.raises(ValueError, match="empty prefix"):
        await summarize(client, "m", [])


async def test_summarize_empty_output_raises() -> None:
    client = Client()
    _install_turns(client, [[_chunk(content=""), _chunk(finish_reason="stop")]])
    with pytest.raises(RuntimeError, match="empty output"):
        await summarize(client, "m", [UserMessage(content="hi")])


# ---- compact ----

async def test_compact_returns_none_when_nothing_to_cut() -> None:
    client = Client()
    history: list[Message] = [UserMessage(content="hi")]
    result = await compact(
        history, client=client, model="m", keep_recent_tokens=10_000
    )
    assert result is None


async def test_compact_replaces_prefix_with_summary_message() -> None:
    client = Client()
    _install_turns(
        client,
        [[_chunk(content="## Goal\nbe brief"), _chunk(finish_reason="stop")]],
    )
    history: list[Message] = []
    for i in range(3):
        history.append(UserMessage(content=f"q{i}: " + "x" * 200))
        history.append(AssistantMessage(content=[TextContent(text="a" * 200)]))

    one_turn = count_tokens(history[-2:])
    result = await compact(
        history,
        client=client,
        model="m",
        keep_recent_tokens=one_turn + 5,
    )
    assert result is not None
    new_history, summary_text, cut_idx = result
    assert cut_idx == 4
    assert summary_text == "## Goal\nbe brief"
    # New history: [summary_msg, last_turn_user, last_turn_assistant]
    assert len(new_history) == 3
    summary_msg = new_history[0]
    assert isinstance(summary_msg, UserMessage)
    assert isinstance(summary_msg.content, str)
    assert "be brief" in summary_msg.content
    assert "<summary>" in summary_msg.content
    # Tail kept verbatim
    assert new_history[1] is history[4]
    assert new_history[2] is history[5]


async def test_compact_does_not_mutate_input_history() -> None:
    client = Client()
    _install_turns(
        client, [[_chunk(content="summary"), _chunk(finish_reason="stop")]]
    )
    history: list[Message] = [
        UserMessage(content="x" * 500),
        AssistantMessage(content=[TextContent(text="a" * 500)]),
        UserMessage(content="y" * 200),
        AssistantMessage(content=[TextContent(text="b" * 200)]),
    ]
    original_ids = [id(m) for m in history]
    one_turn = count_tokens(history[-2:])
    await compact(
        history, client=client, model="m", keep_recent_tokens=one_turn + 5
    )
    # The input history list is unchanged
    assert [id(m) for m in history] == original_ids


async def test_compact_propagates_summarize_failure() -> None:
    client = Client()
    # Empty stream → empty output → summarize raises RuntimeError
    _install_turns(client, [[_chunk(finish_reason="stop")]])
    history: list[Message] = []
    for i in range(3):
        history.append(UserMessage(content=f"q{i}: " + "x" * 200))
        history.append(AssistantMessage(content=[TextContent(text="a" * 200)]))

    one_turn = count_tokens(history[-2:])
    with pytest.raises(RuntimeError):
        await compact(
            history,
            client=client,
            model="m",
            keep_recent_tokens=one_turn + 5,
        )
