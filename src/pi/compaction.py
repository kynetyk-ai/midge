"""Context compaction: summarize old turns into a synthetic user message.

The algorithm walks the history backwards from the end, finds the largest
suffix that fits in `keep_recent_tokens`, and summarizes everything before
that cut. Cuts are only made at `UserMessage` boundaries so a tool call and
its matching tool result never get split across the summary boundary.

The summary is produced by a separate LLM call using `Client.stream` with a
dedicated system prompt and is written into the new history as a single
`UserMessage` whose content wraps the summary in a `<summary>` block — the
agent treats this as just another user turn on the next iteration.

Token counting uses a cheap bytes/4 estimator (no `tiktoken` dependency).
The estimator inflates counts somewhat compared to real tokenization, so set
thresholds with that in mind. Swap in `tiktoken` later if accuracy matters.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from pi.client import Client, Error, TextDelta
from pi.messages import Message, UserMessage

SUMMARIZATION_SYSTEM_PROMPT = """You are a context summarization assistant. Your job is to produce a compact summary of a conversation history so the rest of the conversation can resume with full context but reduced token usage. Do not continue the conversation; only summarize.

Use this EXACT markdown format. Be concise but specific.

## Goal
The user's overall goal in this session.

## Constraints & Preferences
Any explicit constraints, style preferences, or "do not" rules.

## Progress
### Done
What has been completed.
### In Progress
What is partially complete or being investigated.
### Blocked
What is blocked and why.

## Key Decisions
Decisions made and their rationale (especially ones that should not be revisited).

## Next Steps
What the agent should do next when the conversation resumes.

## Critical Context
File paths read or modified, important findings, identifiers, and anything that would be expensive to rediscover."""

SUMMARIZE_INSTRUCTION = (
    "Produce a summary of the conversation above using the exact format "
    "described in your system instructions."
)

COMPACTION_PREFIX = (
    "The conversation history before this point was compacted into the "
    "following summary:\n\n<summary>\n"
)
COMPACTION_SUFFIX = "\n</summary>"


CountTokensFn = Callable[[Sequence[Message]], int]


def count_tokens(messages: Sequence[Message]) -> int:
    """Cheap token estimator: serialize to JSON, divide bytes by 4.

    Inflates counts vs. real tokenization (JSON adds structure) but is
    deterministic and dependency-free. For accurate counts, swap this for
    a tiktoken-based implementation.
    """
    if not messages:
        return 0
    payload = json.dumps(
        [m.model_dump(mode="json") for m in messages], ensure_ascii=False
    )
    return len(payload.encode("utf-8")) // 4


def needs_compaction(
    history: Sequence[Message],
    *,
    threshold_tokens: int,
    count_tokens_fn: CountTokensFn = count_tokens,
) -> bool:
    return count_tokens_fn(history) > threshold_tokens


def find_cut_index(
    history: Sequence[Message],
    *,
    keep_recent_tokens: int,
    count_tokens_fn: CountTokensFn = count_tokens,
) -> int:
    """Return idx such that history[:idx] should be summarized and history[idx:]
    kept. The cut snaps to a UserMessage boundary (so tool call/result pairs
    are never split). Returns 0 to signal "no compaction needed/possible":

    - empty history
    - no UserMessage in history
    - the entire history fits within keep_recent_tokens (best cut would be 0)
    - the only UserMessage is at index 0

    If even the suffix from the latest UserMessage exceeds the budget, that
    suffix is kept anyway — we never drop the most recent turn.
    """
    if not history:
        return 0

    user_indices = [i for i, m in enumerate(history) if isinstance(m, UserMessage)]
    if not user_indices:
        return 0

    latest = user_indices[-1]
    if count_tokens_fn(history[latest:]) > keep_recent_tokens:
        return latest if latest > 0 else 0

    best = latest
    for idx in reversed(user_indices[:-1]):
        if count_tokens_fn(history[idx:]) <= keep_recent_tokens:
            best = idx
        else:
            break

    return best if best > 0 else 0


def make_summary_message(summary_text: str) -> UserMessage:
    return UserMessage(content=COMPACTION_PREFIX + summary_text + COMPACTION_SUFFIX)


async def summarize(
    client: Client,
    model: str,
    prefix_messages: Sequence[Message],
) -> str:
    """Run a one-shot LLM call to summarize the prefix history. Raises on
    error. Returns the concatenated text deltas, stripped.
    """
    if not prefix_messages:
        raise ValueError("Cannot summarize an empty prefix")

    summary_input: list[Message] = [
        *prefix_messages,
        UserMessage(content=SUMMARIZE_INSTRUCTION),
    ]
    text_parts: list[str] = []
    async for ev in client.stream(
        messages=summary_input,
        model=model,
        system=SUMMARIZATION_SYSTEM_PROMPT,
    ):
        if isinstance(ev, TextDelta):
            text_parts.append(ev.delta)
        elif isinstance(ev, Error):
            raise RuntimeError(
                f"summarization failed: {ev.message.error_message or 'unknown error'}"
            )

    text = "".join(text_parts).strip()
    if not text:
        raise RuntimeError("summarization produced empty output")
    return text


async def compact(
    history: Sequence[Message],
    *,
    client: Client,
    model: str,
    keep_recent_tokens: int,
    count_tokens_fn: CountTokensFn = count_tokens,
) -> tuple[list[Message], str, int] | None:
    """Compact `history` if a valid cut exists.

    Returns `(new_history, summary_text, cut_index)` on success, or `None` if
    no compaction was applied (history fits, or no valid cut point).

    The new history is `[summary_message, *history[cut_index:]]`. The caller
    is responsible for swapping the agent's history and (optionally) recording
    the compaction in a session file.
    """
    cut_idx = find_cut_index(
        history,
        keep_recent_tokens=keep_recent_tokens,
        count_tokens_fn=count_tokens_fn,
    )
    if cut_idx == 0:
        return None

    summary_text = await summarize(client, model, history[:cut_idx])
    new_history: list[Message] = [make_summary_message(summary_text), *history[cut_idx:]]
    return new_history, summary_text, cut_idx
