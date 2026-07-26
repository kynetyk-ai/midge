# Context compaction — patterns to borrow from `pi-mono`

Source:
- `pi-mono/packages/coding-agent/src/core/agent-session.ts:1754–1832` — trigger (`_checkCompaction`)
- `pi-mono/packages/coding-agent/src/core/compaction/compaction.ts` — algorithm + prompts
- `pi-mono/packages/coding-agent/src/core/messages.ts:11–17, 176–183` — replacement message shape

## Trigger

Pi-mono fires compaction **after each agent turn completes** (after `agent_end`). Two trigger conditions:

1. **Overflow** — the LLM call for the next turn returned a context-overflow error. Compaction retries the turn once; second failure surfaces an error.
2. **Threshold** — `contextTokens > contextWindow - reserveTokens`. Computed from the last assistant message's `usage.totalTokens` (or summed from input/output/cache_read/cache_write if `totalTokens` is missing).

Defaults from `compaction.ts:121–125`:
- `reserveTokens = 16384` — safety buffer between threshold and the actual context-window limit.
- `keepRecentTokens = 20000` — budget for messages kept verbatim after the cut.

**Note: pi-mono's threshold is absolute, not percentage-based.** Our roadmap mentions "~70% of context" — that's the *spirit* (we should compact well before the wall) but we have flexibility on shape. A fixed token reserve is what pi-mono ships and it's simpler to reason about.

The system prompt and tool definitions are **excluded** from the count. Only message-history tokens count toward the trigger.

## Cut-point algorithm

The compaction routine walks **backwards** from session end, accumulating estimated tokens until it hits `keepRecentTokens`. It then snaps to the nearest valid cut point.

**Valid cut points** (`compaction.ts:299–337`):
- A `user` message — clean turn boundary.
- An `assistant` message — but only if a corresponding tool-result tail follows it on the *kept* side. The kept side then includes that assistant + all matching tool results.
- **Never** a `toolResult` message — it would be orphaned from its tool call.

If the cut lands mid-turn (at an assistant message instead of a user message), pi-mono flags `isSplitTurn = true` and additionally generates a *turn-prefix summary* — a separate small summary of the partial turn — so the kept suffix has its own context.

For our v1, a simpler rule is fine:
- **Cut only at user-message boundaries.** If the budget overshoots a turn, just keep that turn whole. No split-turn handling, no turn-prefix summary.
- This loses a little efficiency (we keep more than `keepRecentTokens` sometimes) but eliminates a class of edge cases.

## The summary call

A separate LLM call with the *same* model. `compaction.ts:530–590`. `max_tokens` is capped at 80% of `reserveTokens` (~13k by default).

The prompt is a structured markdown template (`compaction.ts:454–485`). Verbatim section structure:

```
## Goal
## Constraints & Preferences
## Progress
### Done / In Progress / Blocked
## Key Decisions
## Next Steps
## Critical Context
```

There's also an `UPDATE_SUMMARIZATION_PROMPT` for iterative compactions that preserves the prior summary and merges new work in. For v1 we treat each compaction as a fresh summary — feeding the prior compaction-summary message (already in history) plus the to-be-compacted suffix into the summary call. That's still iterative in effect.

The `SUMMARIZATION_SYSTEM_PROMPT` (`compaction.ts:576`) replaces the agent's system prompt for the summary call so the model doesn't try to "continue the work" — it just summarizes.

## Replacement shape

The summary lives in a synthetic message. In pi-mono it's an internal type `compactionSummary`; on the LLM-conversion boundary it's wrapped as a user message:

```
The conversation history before this point was compacted into the following summary:

<summary>
{summary text}
</summary>
```

(`messages.ts:11–17`.)

For our port, since we don't have a custom internal type system yet, **the simplest faithful adaptation is a `UserMessage` whose content is exactly that wrapper text**. It round-trips through the existing message types, the agent loop sees it as plain user input, and the conversation can resume.

Future: when we want to render the wrapper distinctly in the HTML transcript, we can add a `compaction_summary: bool` flag (in `extra`) or a fresh message type. Defer.

## Atomicity guarantees

The cut algorithm guarantees no tool call/result pair is split. The compaction-replacement happens **between** turns, so:
- All messages strictly before the cut → replaced by the summary.
- All messages from the cut onward → kept verbatim.

Implementation: produce a new history list `[summary_msg] + history[cut_index:]`, swap atomically, then continue.

## Edge cases & guardrails

1. **Anti-thrash guard.** Pi-mono (`agent-session.ts:1773–1778`) skips compaction if the assistant message that triggered it predates the latest compaction boundary — prevents a stale overflow from re-firing immediately. We can mirror with a "last_compacted_at" turn counter and only allow at most one compaction per turn.

2. **Summary-call failure.** Pi-mono emits an error event and **does not modify history**. The session stays in its pre-compaction state. We adopt the same: try-catch the summary call; on failure, log + emit a user-visible warning, keep history intact, let the next turn potentially overflow.

3. **Single overflow retry.** Pi-mono retries a turn once after an overflow-driven compaction. Our v1 can skip this — let the next user turn naturally trigger threshold-based compaction.

4. **Summary larger than threshold.** No explicit guardrail. In practice `keepRecentTokens` + `reserveTokens` (~36k) gives ample headroom on a 200k context, so the summary itself rarely pushes over. We accept the same limitation.

5. **Streaming during compaction.** Pi-mono's loop is synchronous around `_checkCompaction`. Streaming doesn't happen during the summary call — the user just waits. We do the same.

## Token counting

Pi-mono uses `usage.totalTokens` from the LLM's own response when available, falling back to estimation. The OpenAI Chat Completions API does return `usage` on streaming responses (with `stream_options.include_usage = True`), but it's optional and not all OpenAI-compat servers support it.

For v1: use `usage` if available (stash on `AssistantMessage.extra["usage"]` from the streaming chunks); otherwise use a cheap estimator like `len(json.dumps(messages)) // 4` as a fallback. `tiktoken` is more accurate but adds a dependency we don't yet need.

## What Phase 3 implements

In `src/pym/compaction.py`:

```python
def needs_compaction(history, *, threshold_tokens, count_tokens) -> bool: ...

def find_cut_index(history, *, keep_recent_tokens, count_tokens) -> int:
    """Index such that history[:idx] gets summarized and history[idx:] is kept.
    Snaps to user-message boundaries; returns 0 if no valid cut."""

async def summarize(
    client, model, prefix_messages, *,
    system_prompt: str = SUMMARIZATION_SYSTEM_PROMPT,
    user_prompt: str = SUMMARIZE_INSTRUCTION,
) -> str: ...

async def compact(
    history, *, client, model, threshold_tokens, keep_recent_tokens, count_tokens,
) -> list[Message]:
    """Run the full compaction. Returns a NEW history. Raises on summary failure."""
```

In `Agent`:
- After each turn, call `compact(...)` if `needs_compaction(...)`.
- Track a `_last_compaction_turn` counter to prevent re-firing on the same turn.

Tests:
- `find_cut_index` lands on user boundaries, not tool-result mid-pairs.
- Compaction replaces the prefix with one synthetic user message containing the summary wrapper text.
- Threshold trigger fires; below-threshold doesn't.
- Failed summary call leaves history unchanged.
- Anti-thrash: two consecutive turns with same compaction trigger only fires once.
