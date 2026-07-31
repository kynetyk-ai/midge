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
on every token is laggy).

The control surface — compact, clear, reload, switch model, switch profile — is
not reimplemented here. It is `midge.commands.Controls`, the same object the RPC
server drives, and both surfaces enumerate the same `BUILTIN_COMMANDS`. That is
what keeps them from drifting: a command added to that table appears in the
palette without this file being edited.

Two ways to reach it, one table behind both. **Ctrl+P** opens Textual's command
palette, which offers the commands that need no argument and the two whose valid
values are knowable — `set_model` and `use_profile` come with enums, so they
become sub-entries rather than a prompt for free text. **A leading slash** in the
input box does the same and can carry an argument.

A slash only intercepts when the word after it is a command anyone could invoke.
`/etc/hosts is missing` is a sentence, and treating it as a failed command would
make the input box refuse ordinary English about paths.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import VerticalScroll
from textual.logging import TextualHandler
from textual.message import Message
from textual.widgets import Footer, Header, OptionList, Static, TextArea
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerState

from midge.agent import Agent, AgentEnd, SteeringQueue, ToolExecutionEnd, ToolExecutionStart
from midge.client import (
    Done,
    Error,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
)
from midge.commands import BUILTIN_COMMANDS, Controls, Refused
from midge.compaction import compact, needs_compaction
from midge.messages import TextContent, ToolCall
from midge.persistence import Session

_logger = logging.getLogger(__name__)


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
    """A one-line note about what just happened, rendered literally.

    `markup=False` because everything that lands here is someone else's text —
    a model id, a path, a provider's error message — and Textual reads square
    brackets as style tags. `[model is now gpt-4o]` parses as a tag and renders
    as nothing at all, which is the worst way for a status line to fail.
    """

    DEFAULT_CSS = """
    StatusLine { color: $text-muted; padding: 0 1; }
    """

    def __init__(self, content: str) -> None:
        super().__init__(content, markup=False)


class MidgeCommands(Provider):
    """The palette, over the same table the RPC server enumerates."""

    @property
    def _app(self) -> PiApp:
        app = self.app
        assert isinstance(app, PiApp)
        return app

    def _entries(self) -> list[tuple[str, str, str | None]]:
        """(display, description, argument) for everything the palette offers.

        Read off the schema rather than a list kept here, so a command becomes
        palette-invocable the moment its arguments are knowable — and cannot be
        offered before that. Three cases:

        - **No required argument** — one entry that fires it. `compact`.
        - **One required argument with an enum** — one entry per value, because
          a palette is a list you filter, so `set_model gpt-4o` is a thing to
          find rather than a prompt for free text.
        - **Anything else** — omitted. A path or a prompt has to be typed, and
          the slash form is where you can type it.

        The third case is why this is derived: `set_model` has an enum only once
        a `[models]` table exists. With an empty registry it is *not* knowable,
        and an entry firing it with no value would set the model to "".
        """
        app = self._app
        out: list[tuple[str, str, str | None]] = []
        for command in BUILTIN_COMMANDS:
            schema = app.controls.builtin_schema(command)
            properties = schema.get("properties", {})
            required = list(schema.get("required", ()))
            if not required:
                out.append((command.name, command.description, None))
                continue
            if len(required) == 1 and (values := properties[required[0]].get("enum")):
                out.extend(
                    (f"{command.name} {v}", command.description, v) for v in values
                )
        for skill in app.controls.skills:
            out.append((f"skill:{skill.name}", skill.description, None))
        return out

    async def discover(self) -> Hits:
        for display, description, argument in self._entries():
            yield DiscoveryHit(
                display, partial(self._app.invoke, display, argument), help=description
            )

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for display, description, argument in self._entries():
            if (score := matcher.match(display)) > 0:
                yield Hit(
                    score,
                    matcher.highlight(display),
                    partial(self._app.invoke, display, argument),
                    help=description,
                )


class Sidebar(VerticalScroll):
    """What the agent *is*, and what it could be instead.

    The palette is verbs — compact, clear, reload, things you do once. This is
    nouns: the session, the profile and the model are each a set of named
    alternatives with exactly one current, which is a different shape and wants
    a different affordance. `Controls` already reports all three that way, so
    this renders rather than computes.

    It also answers a question the TUI could not previously answer at all. A
    modal shows you the list only while you are choosing and then vanishes;
    docked, it shows what you are on. Before this the only visible state was the
    model in the title bar.

    A section with nothing to offer is omitted rather than shown empty — the
    same rule the palette follows, and the reason the model section disappears
    without a `[models]` table: with an empty registry midge cannot know what
    the alternatives are.
    """

    DEFAULT_CSS = """
    Sidebar { dock: left; width: 34; border-right: solid $accent; padding: 0 1; }
    Sidebar.hidden { display: none; }
    Sidebar > .section { color: $text-muted; text-style: bold; padding: 1 0 0 0; }
    Sidebar > OptionList { border: none; background: transparent; height: auto; }
    """

    def rebuild(self, controls: Controls) -> None:
        """Read the current state and redraw. Called on every open.

        Cheaper than staying in sync: sessions appear on disk from other
        processes, and a profile switch changes two sections at once.
        """
        self.remove_children()
        sections = 0
        for title, options in (
            ("sessions", _session_options(controls)),
            ("profiles", _profile_options(controls)),
            ("model", _model_options(controls)),
        ):
            if not options:
                continue
            sections += 1
            self.mount(Static(title, classes="section"))
            self.mount(OptionList(*options))
        if not sections:
            self.mount(Static("nothing to switch to", classes="section"))


# NUL, because the argument half is a filesystem path and everything printable
# can appear in one.
_ARG = "\x00"
_CURRENT = "\u25cf"


def _mark(label: str, *, current: bool) -> str:
    return f"{_CURRENT if current else ' '} {label}"


def _session_options(controls: Controls) -> list[Option]:
    return [
        Option(
            _mark(s["name"] or Path(s["path"]).name, current=bool(s["current"])),
            id=f"open_session{_ARG}{s['path']}",
        )
        for s in controls.session_list()
    ]


def _profile_options(controls: Controls) -> list[Option]:
    return [
        Option(_mark(name, current=name == controls.profile), id=f"use_profile{_ARG}{name}")
        for name in controls.profiles.names()
    ]


def _model_options(controls: Controls) -> list[Option]:
    registry = controls.agent.client.registry
    return [
        Option(_mark(name, current=name == controls.agent.model), id=f"set_model{_ARG}{name}")
        for name in (registry.names() if registry else ())
    ]


class PiApp(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #log { height: 1fr; padding: 0 1; }
    #input { height: 6; border-top: solid $accent; }
    """

    COMMANDS: ClassVar[set[type[Provider] | Callable[[], type[Provider]]]] = {MidgeCommands}

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "interrupt", "Interrupt", priority=True),
        Binding("ctrl+d", "quit", "Quit", priority=True),
        Binding("ctrl+b", "toggle_sidebar", "Switch to…", priority=True),
        Binding("escape", "clear_input", "Clear input"),
    ]

    def __init__(
        self,
        controls: Controls,
        *,
        compaction_threshold: int | None = None,
    ) -> None:
        super().__init__()
        self.controls = controls
        # The threshold stays here rather than on `Controls`: automatic
        # compaction is this interface deciding when to act, not an operation
        # anyone invokes. `keep_recent` is shared, because it is the same
        # summary either way.
        self.compaction_threshold = compaction_threshold
        controls.runner = self
        # Steering has to be a real queue before a turn starts, or a message
        # typed mid-turn has nowhere to land.
        if controls.agent.steering is None:
            controls.agent.steering = SteeringQueue()
        self._current_assistant: AssistantBubble | None = None
        self._tool_bubbles: dict[str, ToolCallBubble] = {}
        self._current_worker: Worker[None] | None = None
        self.title = f"midge · {controls.agent.model}"

    @property
    def agent(self) -> Agent:
        return self.controls.agent

    @property
    def session(self) -> Session | None:
        return self.controls.session

    @property
    def compaction_keep_recent(self) -> int:
        return self.controls.compaction_keep_recent

    # `Controls.Runner`: what "a run is in flight" means here is a worker.
    def busy(self) -> bool:
        return self._current_worker is not None and self._current_worker.state is WorkerState.RUNNING

    def cancel(self) -> None:
        if self._current_worker is not None:
            self._current_worker.cancel()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Sidebar(id="sidebar", classes="hidden")
        yield VerticalScroll(id="log")
        yield _SubmitTextArea(id="input", soft_wrap=True)
        yield Footer()

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar", Sidebar)
        if sidebar.has_class("hidden"):
            sidebar.rebuild(self.controls)
            sidebar.remove_class("hidden")
            # Focus follows, or the arrow keys would still be editing the draft.
            lists = sidebar.query(OptionList)
            if lists:
                lists.first().focus()
        else:
            self._close_sidebar()

    def _close_sidebar(self) -> None:
        self.query_one("#sidebar", Sidebar).add_class("hidden")
        self.query_one("#input", _SubmitTextArea).focus()

    @on(OptionList.OptionSelected)
    async def _on_sidebar_choice(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is None:
            return
        name, _, argument = event.option_id.partition(_ARG)
        # Closed first: the choice changes what the panel would show, and
        # leaving it open displaying the old state is worse than dismissing it.
        self._close_sidebar()
        await self.invoke(name, argument)

    def on_mount(self) -> None:
        self.query_one("#input", _SubmitTextArea).focus()
        if self.agent.history:
            log = self.query_one("#log", VerticalScroll)
            log.mount(StatusLine(f"[resumed: {len(self.agent.history)} prior messages]"))

    def _as_command(self, text: str) -> tuple[str, str] | None:
        """`(name, argument)` if this is a command, else None.

        Only a name from the table intercepts. Anything else starting with a
        slash is a sentence about a path, and refusing it as an unknown command
        would make the input box reject ordinary English.
        """
        if not text.startswith("/"):
            return None
        name, _, argument = text[1:].partition(" ")
        known = {c.name for c in BUILTIN_COMMANDS}
        if name in known:
            return name, argument.strip()
        if name.startswith("skill:") and any(
            s.name == name[len("skill:") :] for s in self.controls.skills
        ):
            return name, argument.strip()
        return None

    @on(_SubmitTextArea.Submitted)
    async def _on_submit(self, message: _SubmitTextArea.Submitted) -> None:
        command = self._as_command(message.value)
        if command is not None:
            await self.invoke(f"{command[0]} {command[1]}".strip(), None)
            return

        if self.busy():
            # Queued, not cancel-and-restart. The old path cancelled the worker
            # and awaited its teardown so two writers could not interleave into
            # `Agent.history` — steering removes the second writer instead, and
            # keeps the work already done in the turn.
            steering = self.agent.steering
            assert steering is not None, "steering is created in __init__"
            steering.steer(self.controls.expand(message.value))
            log = self.query_one("#log", VerticalScroll)
            await log.mount(StatusLine(f"[queued: {message.value}]"))
            log.scroll_end(animate=False)
            return

        self._current_worker = self.run_worker(
            self._run_turn(message.value),
            exclusive=True,
            exit_on_error=False,
        )

    async def invoke(self, display: str, argument: str | None) -> None:
        """Run one command and say what happened.

        `display` is `name` or `name argument` — the palette builds it that way
        so an entry is a thing you can read, and the slash path produces the
        same string. A skill is not a command: it expands to a prompt, which is
        the `invoke: "prompt"` the command table already declares.
        """
        name, _, inline = display.partition(" ")
        value = argument if argument is not None else inline.strip()
        log = self.query_one("#log", VerticalScroll)

        if name.startswith("skill:"):
            self._current_worker = self.run_worker(
                self._run_turn(f"/{display}"), exclusive=True, exit_on_error=False
            )
            return

        try:
            result = await self._dispatch(name, value)
        except Refused as e:
            await log.mount(StatusLine(f"[{name}: {e}]"))
        except (OSError, ValueError) as e:
            _logger.exception("tui_command_failed command=%s", name)
            await log.mount(StatusLine(f"[{name} failed: {e}]"))
        else:
            await log.mount(StatusLine(f"[{result}]"))
            self.title = f"midge · {self.agent.model}"
        log.scroll_end(animate=False)

    async def _dispatch(self, name: str, value: str) -> str:
        """Call the operation and describe it. Refusals propagate."""
        c = self.controls
        # A command whose argument is required cannot run without one. The
        # palette never offers these bare; a slash can be typed bare.
        if not value and name in {
            "set_model",
            "use_profile",
            "set_session_name",
            "set_system_prompt",
            "new_session",
            "open_session",
        }:
            raise Refused(f"{name} needs an argument")
        match name:
            case "abort":
                dropped = c.abort()
                return f"aborted; {len(dropped)} queued message(s) dropped"
            case "compact":
                data = await c.compact()
                if data["summary"] is None:
                    return "nothing to compact"
                return (
                    f"compacted: {data['cut_index']} messages summarized; "
                    f"history is now {data['message_count']} messages"
                )
            case "clear_context":
                return f"cleared {c.clear_context()['cleared']} messages"
            case "reload":
                data = await c.reload()
                return (
                    f"reloaded {', '.join(data['targets']) or 'nothing'} — "
                    f"{data['tools']} tools, {data['skills']} skills"
                )
            case "set_model":
                c.set_model(value)
                return f"model is now {value}"
            case "use_profile":
                data = c.use_profile(value)
                return f"profile {data['profile']} — {len(data['tools'])} tools, {data['model']}"
            case "set_session_name":
                return f"session named {c.set_session_name(value)['name']}"
            case "set_system_prompt":
                c.set_system_prompt(value)
                return "system prompt replaced"
            case "new_session":
                return f"recording to {c.new_session(Path(value))['session']}"
            case "open_session":
                data = c.open_session(Path(value))
                return f"opened {data['session']} — {data['messages']} messages"
        raise Refused(f"unknown command {name!r}")

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
                _logger.exception("compaction_failed")
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
                bubble.update(f"⚙ {ev.tool_call.name}({ev.tool_call.arguments}) — running...")
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

    @on(Worker.StateChanged)
    def _on_worker_state(self, event: Worker.StateChanged) -> None:
        # `exit_on_error=False` otherwise swallows the exception into Textual's
        # internal log and the turn just stops mid-render with no explanation.
        if event.state is not WorkerState.ERROR:
            return
        err = event.worker.error
        _logger.error("tui_turn_failed error=%s", type(err).__name__, exc_info=err)
        log = self.query_one("#log", VerticalScroll)
        log.mount(StatusLine(f"[turn failed: {type(err).__name__}: {err}]"))
        log.scroll_end(animate=False)

    def action_interrupt(self) -> None:
        worker = self._current_worker
        if worker is not None and worker.state == WorkerState.RUNNING:
            worker.cancel()

    def action_clear_input(self) -> None:
        if not self.query_one("#sidebar", Sidebar).has_class("hidden"):
            self._close_sidebar()
            return
        self.query_one("#input", _SubmitTextArea).clear()


def tui_log_handler(log_file: Path | None = None) -> logging.Handler | None:
    """A log handler safe to install before `App.run()`.

    `logging.StreamHandler` binds `sys.stderr` at construction, so it writes
    straight past Textual's `redirect_stderr` and shreds the display.
    `TextualHandler` resolves the active app per record instead — but it routes
    to the devtools console, visible only under `textual console`, so a log file
    is the practical way to read these. `None` hands that case back to
    `logs.configure`, which opens the file itself.
    """
    return None if log_file else TextualHandler()


def run_tui(controls: Controls, *, compaction_threshold: int | None = None) -> None:
    PiApp(controls, compaction_threshold=compaction_threshold).run()
