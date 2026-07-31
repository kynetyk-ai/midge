"""Hooks for working inside the midge repo itself.

    midge --extension-dir .agents/extensions

**Extensions are not auto-discovered.** Skills are — `.agents/skills/` and
`.midge/skills/` are on `default_skill_dirs()` — but an extension is arbitrary
Python that runs in-process, so it is loaded only when a flag names it. That
asymmetry is deliberate: finding a `SKILL.md` on disk and reading it is not the
same risk as importing a file.

One handler of each kind, because that is the distinction worth learning:

    observe()             sees everything, changes nothing, cannot block
    on("tool_call")       decides — the only point that stops work happening
    on("context")         transforms — chained, each sees the last one's output

`register_hooks` is what the loader looks for. It is called with a source-tagged
view of the shared `Hooks`, so a handler that raises names this file rather than
appearing from nowhere. `add_cleanup` runs on unload, which is what makes
`reload` safe to press twice.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# The loader injects a `midge.ext.repo_guard` logger; this is the fallback for
# when the module is imported directly. Never `print()` from an extension —
# under the TUI, Textual drops stdout, and in RPC mode stdout is the protocol.
log: logging.Logger = logging.getLogger("midge.ext.repo_guard")

SYSTEM_PROMPT = (
    "You are working inside the midge repository. Runtime state under `.midge/` "
    "is gitignored: never write there, and never commit it. The core harness is "
    "size-capped — run `poetry run python scripts/loc.py` before proposing a "
    "large addition to `src/midge/*.py`."
)

# Written by a running agent, gitignored, and therefore invisible in review. A
# transcript left here once made the test suite read whatever sessions happened
# to be on the machine.
_RUNTIME_DIRS = (".midge/", ".midge-")


def register_hooks(hooks: Any) -> None:
    hooks.on("tool_call", _refuse_writes_to_runtime_state)
    hooks.on("context", _note_the_budget)
    hooks.observe(_audit)
    hooks.add_cleanup(_on_unload)


def _refuse_writes_to_runtime_state(event: Any, ctx: Any) -> Any:
    """Decide: block a write into gitignored runtime state.

    Returning `ToolCallResult(block=True)` is the only way to stop work before
    it happens. `observe` cannot — it is told, not asked.
    """
    from midge.hooks import ToolCallResult

    name = event.tool_call.name
    if name not in ("write", "edit"):
        return None
    target = str(event.tool_call.arguments.get("path", ""))
    if not any(marker in target for marker in _RUNTIME_DIRS):
        return None
    log.warning("repo_guard_write_blocked path=%s", target)
    return ToolCallResult(
        block=True,
        reason=(
            f"{target} is gitignored runtime state. Write somewhere tracked, or "
            "use a temporary directory if this is scratch work."
        ),
    )


def _note_the_budget(event: Any, ctx: Any) -> Any:
    """Transform: append the current size-budget headroom to the context.

    Chained — the messages returned here are what the next handler sees, and
    what finally reaches the provider. Returning `None` leaves them untouched,
    which is what this does whenever the number is not interesting.
    """
    from midge.hooks import ContextResult
    from midge.messages import UserMessage

    headroom = _core_headroom()
    if headroom is None or headroom > 500:
        return None
    log.info("repo_guard_budget_tight headroom=%d", headroom)
    return ContextResult(
        messages=[
            *event.messages,
            UserMessage(
                content=(
                    f"[repo] The core size budget has {headroom} lines of "
                    "headroom. Prefer simplifying over adding, and put anything "
                    "that is not core into tools/ or an extension."
                )
            ),
        ]
    )


def _audit(event: Any, ctx: Any) -> None:
    """Observe: told about every event, changes nothing.

    The catch-all is where counting and logging belong. It cannot block and its
    return value is ignored, which is what makes it safe to add anywhere.
    """
    if event.type == "tool_call":
        log.info("repo_guard_tool tool=%s", event.tool_call.name)
    elif event.type == "before_compact":
        log.info("repo_guard_compacting messages=%d", len(event.history))
    elif event.type in ("session_start", "session_end"):
        log.info("repo_guard_%s path=%s", event.type, event.path or "-")


def _on_unload() -> None:
    """Runs when the extension is unloaded, which `reload` does before re-importing.

    Without this, hooks registered by a re-imported module would stack on top of
    the old ones and every handler would fire twice.
    """
    log.info("repo_guard_unloaded")


def _core_headroom() -> int | None:
    """Lines left under the 5k cap, or None if the counter is not importable."""
    import sys

    root = Path(__file__).resolve().parent.parent.parent
    scripts = root / "scripts"
    if not (scripts / "loc.py").exists():
        return None
    sys.path.insert(0, str(scripts))
    try:
        import loc  # type: ignore[import-not-found]

        counted = sum(loc.count(p) for p in sorted(loc.CORE.glob("*.py")))
        return loc.BUDGET - counted
    except Exception as e:
        log.warning("repo_guard_budget_unavailable error=%s", e)
        return None
    finally:
        sys.path.remove(str(scripts))
