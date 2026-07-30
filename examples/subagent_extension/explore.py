"""Sub-agent extension: a read-only explorer the main agent can delegate to.

    poetry run python -m examples.coding_agent \\
        --extension-dir examples/subagent_extension "where is the cut point chosen?"

`@subagent` returns a `Tool`, so the ordinary extension loader finds this file
and the model sees a `spawn_explore` tool alongside `read`, `bash`, and the
rest. No separate agent directory, no new flag.

The point of declaring the inputs here rather than accepting a free-form task
string is that the model's choices are bounded: it supplies a question and
optional paths, and nothing else about the child is up to it — not the system
prompt, not the tool list, not the model.

Delegating this work keeps a long search out of the main conversation. The
parent gets the findings; the dozens of reads that produced them stay in the
sub-agent's own transcript.
"""

from __future__ import annotations

from midge.subagents import subagent

EXPLORE_PROMPT = (
    "You are a code explorer. Answer the question by reading the repository, "
    "then report what you found and where.\n\n"
    "Cite concrete `path:line` references for every claim — the caller cannot "
    "see what you read, so an unreferenced assertion is useless to them.\n\n"
    "You are read-only. You have no tools that modify anything, so do not "
    "propose edits or claim to have made them. If the answer is not in the "
    "repository, say so plainly rather than guessing.\n\n"
    "Be brief. Your entire reply is what the caller receives."
)


@subagent(
    description=(
        "Delegate a codebase search to a read-only explorer and get back a "
        "summary with path:line references. Use this when answering would mean "
        "reading many files and only the conclusion matters — it keeps the "
        "search out of this conversation. Not for edits: the explorer cannot "
        "change anything."
    ),
    prompt=EXPLORE_PROMPT,
    tools=("read", "bash"),
    # The budget for an ordinary search. A caller can ask for more, up to
    # `[subagents] max_timeout`, because `timeout` is in the signature below.
    timeout=180,
)
async def explore(
    question: str,
    paths: list[str] | None = None,
    timeout: float | None = None,
) -> str:
    """Compose the explorer's opening message from validated inputs.

    `timeout` is declared here and so appears in the tool schema — that is the
    whole opt-in. The model may ask for longer on a big search; it cannot ask
    for longer than the operator's ceiling, and an author who wants a fixed
    budget just leaves the parameter out.
    """
    scope = "\n".join(f"- {p}" for p in paths) if paths else "- (the whole repository)"
    return f"Question: {question}\n\nStart from:\n{scope}"
