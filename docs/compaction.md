# Compaction

Compaction replaces the older part of the conversation with a model-written summary so a long session fits in the context window. The code is [`src/midge/compaction.py`](../src/midge/compaction.py); `Controls` in [`src/midge/commands.py`](../src/midge/commands.py) decides when it runs.

## When it runs

- **Automatically**, after a turn, when `[compaction] threshold` (or `--compaction-threshold`) is set and the measured context exceeds it. It is off when no threshold is set.
- **On request**, through the `compact` command: `Ctrl+P` or `/compact` in the TUI, or the `compact` RPC command. It is refused while a run is in flight.

Compaction never runs during a turn. Summarizing is itself a provider call, and replacing `agent.history` while a turn is appending to it would drop that turn's messages.

In RPC mode automatic compaction is reported as `compaction_start` and `compaction_end` events after the turn's `agent_end` (see [rpc](rpc.md)).

## Measuring the context

`measure_context` starts from the most recent assistant message that completed with `usage`: its `input` plus `output` tokens are the provider's own count of everything sent, system prompt included. Messages appended after it are estimated. With no usage anywhere in history, the whole history is estimated.

The estimate is the UTF-8 length of the messages serialized as JSON, divided by four. It runs higher than real tokenization. Usage is only reported when the provider sends it; for chat-completions servers that depends on `include_usage` (see [providers](providers.md)).

## Choosing the cut

`find_cut_index` keeps the longest recent suffix that fits in `[compaction] keep_recent` (or `--compaction-keep-recent`) tokens, measured with the estimator. The cut falls at a `UserMessage` or `AssistantMessage` as long as no tool call is left open across it. The suffix from the latest cut point is always kept, even when it alone exceeds the budget.

Nothing is compacted when there is no cut point, or when the whole history already fits.

A `before_compact` hook receives the history and the proposed `cut_index`. It can cancel compaction, or return its own `cut_index`, which is moved forward to the next valid cut point; an index that ends up at the start or past the end cancels compaction. See [hooks](hooks.md).

## The summary

`summarize` sends the messages before the cut to the same client and model, with a dedicated system prompt that tells the model to summarize and not continue the work. The summary uses fixed sections: Goal, Constraints & Preferences, Progress (Done, In Progress, Blocked), Key Decisions, Next Steps and Critical Context. A previous summary is part of the prefix, so repeated compactions fold earlier summaries in.

The new history is one `UserMessage` wrapping the summary in a `<summary>` block (`messages.make_summary_message`), followed by the kept suffix. The agent treats it as ordinary user input.

If the summary request fails or returns no text, compaction raises and history is unchanged. After an automatic compaction the error is logged and reported in `compaction_end`; the turn itself has already completed. Cancelling during the summary request cancels compaction the same way.

## Persistence

When a session is open, `Controls` appends a `compaction` record holding the summary and cut index. Loading the session replays it, so a resumed agent holds the compacted history while the file still contains every original message. See [sessions](sessions.md).

## Using it directly

`compact(history, client=..., model=..., keep_recent_tokens=..., hooks=...)` returns `(new_history, summary, cut_index)`, or `None` when nothing was compacted. The caller swaps the history and records the compaction. [`examples/coding_agent.py`](../examples/coding_agent.py) does this after each turn without `Controls`.

Both `needs_compaction` and `find_cut_index` accept a `count_tokens_fn`, so a more accurate tokenizer can replace the estimator.
