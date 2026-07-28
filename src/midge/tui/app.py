"""Textual TUI for interactive use.

A minimum-viable shell:
- Header (model + status), scrolling conversation log, multi-line input box, Footer.
- User text becomes a UserBubble. Assistant text streams into a single
  AssistantBubble that grows in place. Tool calls/executions render as
  inline status cards.
- Ctrl+J submits the input (most terminals send this for Ctrl+Enter).
- Ctrl+C interrupts the current turn (cancels the run worker).
- Ctrl+D quits.
- Esc clears the input draft.

No custom widgets, no markdown rendering during streaming (markdown re-parse
on every token is laggy). Polish later — Phase 4 is just "interactive UI
suitable for daily use".
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Footer, Header, Static, TextArea
from textual.worker import Worker, WorkerState

from midge.agent import Agent, AgentEnd, ToolExecutionEnd, ToolExecutionStart
from midge.client import (
    Done,
    Error,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
)
from midge.compaction import compact, needs_compaction
from midge.messages import TextContent, ToolCall
from midge.persistence import Session


class _SubmitTextArea(TextArea):
    """TextArea where Enter submits and Alt+Enter inserts a newline.

    Ctrl+J is kept as a fallback for terminals that don't deliver a clean
    Enter keysym.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "submit", "Submit", show=False, priority=True),
        Binding("alt+enter", "newline", "Newline", show=True, priority=True),
        Binding("ctrl+j", "submit", "Submit", show=False),
    ]

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def action_submit(self) -> None:
        text = self.text.strip()
        if not text:
            return
        self.post_message(self.Submitted(text))
        self.clear()

    def action_newline(self) -> None:
        self.insert("\n")


class UserBubble(Static):
    DEFAULT_CSS = """
    UserBubble {
        padding: 0 1;
        margin: 1 0 0 0;
        background: $boost;
        border-left: thick $primary;
    }
    """


class AssistantBubble(Static):
    DEFAULT_CSS = """
    AssistantBubble {
        padding: 0 1;
        margin: 1 0 0 0;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._text = ""

    def append(self, delta: str) -> None:
        self._text += delta
        self.update(self._text)


class ToolCallBubble(Static):
    DEFAULT_CSS = """
    ToolCallBubble {
        padding: 0 1;
        margin: 1 0 0 0;
        background: $surface;
        border-left: thick $accent;
    }
    ToolCallBubble.error {
        border-left: thick $error;
    }
    """


class StatusLine(Static):
    DEFAULT_CSS = """
    StatusLine { color: $text-muted; padding: 0 1; }
    """


class PiApp(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #log { height: 1fr; padding: 0 1; }
    #input { height: 6; border-top: solid $accent; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "interrupt", "Interrupt", priority=True),
        Binding("ctrl+d", "quit", "Quit", priority=True),
        Binding("escape", "clear_input", "Clear input"),
    ]

    def __init__(
        self,
        agent: Agent,
        *,
        session: Session | None = None,
        compaction_threshold: int | None = None,
        compaction_keep_recent: int = 20_000,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.session = session
        self.compaction_threshold = compaction_threshold
        self.compaction_keep_recent = compaction_keep_recent
        self._current_assistant: AssistantBubble | None = None
        self._tool_bubbles: dict[str, ToolCallBubble] = {}
        self._current_worker: Worker[None] | None = None
        self.title = f"midge · {agent.model}"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield VerticalScroll(id="log")
        yield _SubmitTextArea(id="input", soft_wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#input", _SubmitTextArea).focus()
        if self.agent.history:
            log = self.query_one("#log", VerticalScroll)
            log.mount(StatusLine(f"[resumed: {len(self.agent.history)} prior messages]"))

    @on(_SubmitTextArea.Submitted)
    def _on_submit(self, message: _SubmitTextArea.Submitted) -> None:
        self._current_worker = self.run_worker(
            self._run_turn(message.value),
            exclusive=True,
            exit_on_error=False,
        )

    async def _run_turn(self, prompt: str) -> None:
        log = self.query_one("#log", VerticalScroll)
        await log.mount(UserBubble(prompt))
        log.scroll_end(animate=False)
        self._current_assistant = None
        self._tool_bubbles = {}

        # `new_messages` never reaches us if the turn is cancelled, so persist
        # the interrupted turn from the history tail instead.
        mark = len(self.agent.history)

        try:
            async for ev in self.agent.stream(prompt):
                self._handle_event(ev, log)
                log.scroll_end(animate=False)
                if isinstance(ev, AgentEnd) and self.session is not None:
                    self.session.append_many(ev.new_messages)
        except asyncio.CancelledError:
            if self.session is not None:
                self.session.append_many(self.agent.history[mark:])
            await log.mount(StatusLine("[interrupted]"))
            log.scroll_end(animate=False)
            raise

        if self.compaction_threshold is not None and needs_compaction(
            self.agent.history, threshold_tokens=self.compaction_threshold
        ):
            await log.mount(StatusLine("[compacting context...]"))
            log.scroll_end(animate=False)
            try:
                result = await compact(
                    self.agent.history,
                    client=self.agent.client,
                    model=self.agent.model,
                    keep_recent_tokens=self.compaction_keep_recent,
                    hooks=self.agent.hooks,
                )
            except Exception as e:
                await log.mount(StatusLine(f"[compaction failed: {e}]"))
                return
            if result is not None:
                new_history, summary_text, cut_idx = result
                self.agent.history = new_history
                if self.session is not None:
                    self.session.append_compaction(summary=summary_text, cut_index=cut_idx)
                await log.mount(
                    StatusLine(
                        f"[compacted: {cut_idx} messages summarized; "
                        f"history is now {len(new_history)} messages]"
                    )
                )

    def _handle_event(self, ev: StreamEvent | Any, log: VerticalScroll) -> None:
        if isinstance(ev, TextDelta):
            if self._current_assistant is None:
                self._current_assistant = AssistantBubble()
                log.mount(self._current_assistant)
            self._current_assistant.append(ev.delta)
        elif isinstance(ev, Done | Error):
            self._current_assistant = None
            if isinstance(ev, Error):
                log.mount(StatusLine(f"[error: {ev.message.error_message}]"))
        elif isinstance(ev, ToolCallStart):
            tc = ev.partial.content[ev.content_index]
            assert isinstance(tc, ToolCall)
            bubble = ToolCallBubble(f"⚙ {tc.name}(...)")
            self._tool_bubbles[tc.id] = bubble
            log.mount(bubble)
        elif isinstance(ev, ToolCallEnd):
            bubble = self._tool_bubbles.get(ev.tool_call.id)
            if bubble is not None:
                bubble.update(f"⚙ {ev.tool_call.name}({ev.tool_call.arguments})")
        elif isinstance(ev, ToolExecutionStart):
            bubble = self._tool_bubbles.get(ev.tool_call.id)
            if bubble is not None:
                bubble.update(
                    f"⚙ {ev.tool_call.name}({ev.tool_call.arguments}) — running..."
                )
        elif isinstance(ev, ToolExecutionEnd):
            bubble = self._tool_bubbles.get(ev.tool_call.id)
            if bubble is None:
                return
            text = ""
            if ev.result.content and isinstance(ev.result.content[0], TextContent):
                text = ev.result.content[0].text
            preview = text if len(text) <= 200 else text[:200] + "…"
            tag = "ERR" if ev.result.is_error else "OK"
            if ev.result.is_error:
                bubble.add_class("error")
            bubble.update(f"⚙ {ev.tool_call.name} → [{tag}] {preview}")
        elif isinstance(ev, AgentEnd):
            self._current_assistant = None

    def action_interrupt(self) -> None:
        worker = self._current_worker
        if worker is not None and worker.state == WorkerState.RUNNING:
            worker.cancel()

    def action_clear_input(self) -> None:
        self.query_one("#input", _SubmitTextArea).clear()


def run_tui(
    agent: Agent,
    *,
    session: Session | None = None,
    compaction_threshold: int | None = None,
    compaction_keep_recent: int = 20_000,
) -> None:
    PiApp(
        agent,
        session=session,
        compaction_threshold=compaction_threshold,
        compaction_keep_recent=compaction_keep_recent,
    ).run()
