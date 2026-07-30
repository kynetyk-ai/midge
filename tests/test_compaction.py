from __future__ import annotations

from typing import Any

import httpx
import openai
import pytest

from midge.client import Client
from midge.compaction import (
    compact,
    count_tokens,
    find_cut_index,
    measure_context,
    needs_compaction,
    summarize,
)
from midge.hooks import CompactResult, Hooks
from midge.messages import (
    COMPACTION_PREFIX,
    COMPACTION_SUFFIX,
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    make_summary_message,
    repair_history,
)
from midge.providers.openai_compat import encode_messages
from tests.fakes import ScriptedProvider, finish, install, say


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
    captured = install(
        client,
        [
            [
                say("## Goal\n"),
                say("learn things"),
                finish(),
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


async def test_summarize_inherits_client_retry() -> None:
    # summarize() calls Client.stream directly, bypassing Agent, so it picks up
    # the provider retry for free. Locked in because it is easy to lose.
    client = Client(retry_base_delay=0)
    calls = [0]
    turns = iter([[say("## Goal\nx"), finish()]])

    async def on_open(body: Any) -> list[Any]:
        calls[0] += 1
        if calls[0] == 1:
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
        return next(turns)

    client.provider = ScriptedProvider(on_open)

    text = await summarize(client, "m", [UserMessage(content="hi")])
    assert calls[0] == 2
    assert text == "## Goal\nx"


async def test_summarize_empty_output_raises() -> None:
    client = Client()
    install(client, [[say(""), finish()]])
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
    install(
        client,
        [[say("## Goal\nbe brief"), finish()]],
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
    install(
        client, [[say("summary"), finish()]]
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
    install(client, [[finish()]])
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


async def test_hook_cut_index_snaps_to_user_boundary() -> None:
    """An arbitrary hook index could split a tool call from its result — issue #33."""
    history: list[Message] = [
        UserMessage(content="one"),
        AssistantMessage(
            content=[ToolCall(id="t1", name="read", arguments={})], stop_reason="tool_use"
        ),
        ToolResultMessage(tool_call_id="t1", tool_name="read", content=[TextContent(text="x")]),
        UserMessage(content="two"),
        AssistantMessage(content=[TextContent(text="done")], stop_reason="stop"),
    ]

    hooks = Hooks()
    # cut_index=2 lands on the tool result, splitting it from its call.
    hooks.on("before_compact", lambda ev, ctx: CompactResult(cut_index=2))

    client = Client()
    install(client, [[say("SUMMARY"), finish()]])

    result = await compact(
        history, client=client, model="gpt-4o", keep_recent_tokens=10, hooks=hooks
    )
    assert result is not None
    new_history, _, cut_idx = result
    assert cut_idx == 3
    assert isinstance(new_history[1], UserMessage)
    wire = encode_messages(repair_history(new_history))
    answered = {m["tool_call_id"] for m in wire if m.get("role") == "tool"}
    requested = {
        tc["id"]
        for m in wire
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    assert requested == answered


# ---- measure_context ----


def _assistant(text: str, *, usage: Usage | None = None, stop: str = "stop") -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        stop_reason=stop,  # type: ignore[arg-type]
        usage=usage,
    )


def test_measure_context_falls_back_to_the_estimator() -> None:
    history: list[Message] = [UserMessage(content="hi"), _assistant("hello")]
    assert measure_context(history) == count_tokens(history)


def test_measure_context_prefers_reported_usage() -> None:
    # The estimator would return a much larger number for this history; the
    # provider's own count wins.
    history: list[Message] = [
        UserMessage(content="x" * 4000),
        _assistant("ok", usage=Usage(input=300, output=12)),
    ]
    assert count_tokens(history) > 1000
    assert measure_context(history) == 312


def test_measure_context_estimates_only_the_tail() -> None:
    tail: list[Message] = [UserMessage(content="y" * 400)]
    history: list[Message] = [
        UserMessage(content="x" * 4000),
        _assistant("ok", usage=Usage(input=300, output=12)),
        *tail,
    ]
    assert measure_context(history) == 312 + count_tokens(tail)


def test_measure_context_ignores_failed_and_aborted_turns() -> None:
    # Their counts describe a request that never completed.
    history: list[Message] = [
        UserMessage(content="a"),
        _assistant("ok", usage=Usage(input=300, output=12)),
        UserMessage(content="b"),
        _assistant("", usage=Usage(input=99_999, output=0), stop="error"),
        _assistant("", usage=Usage(input=88_888, output=0), stop="aborted"),
    ]
    tail = history[2:]
    assert measure_context(history) == 312 + count_tokens(tail)


def test_measure_context_uses_the_most_recent_usage() -> None:
    history: list[Message] = [
        UserMessage(content="a"),
        _assistant("one", usage=Usage(input=100, output=5)),
        UserMessage(content="b"),
        _assistant("two", usage=Usage(input=400, output=9)),
    ]
    assert measure_context(history) == 409


def test_needs_compaction_uses_reported_usage() -> None:
    # bytes/4 over this history is far above 500; the provider says otherwise,
    # and compaction destroys history, so the real number decides.
    history: list[Message] = [
        UserMessage(content="x" * 8000),
        _assistant("ok", usage=Usage(input=400, output=10)),
    ]
    assert count_tokens(history) > 500
    assert needs_compaction(history, threshold_tokens=500) is False
    assert needs_compaction(history, threshold_tokens=300) is True


def test_needs_compaction_respects_an_explicit_counter() -> None:
    history: list[Message] = [
        UserMessage(content="hi"),
        _assistant("ok", usage=Usage(input=1, output=1)),
    ]
    assert needs_compaction(history, threshold_tokens=10, count_tokens_fn=lambda _: 999) is True
